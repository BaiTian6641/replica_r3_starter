# Execution state — integrated memory, cognition and learning

## BF16/Muon and parallel GPU budget follow-up — 26 October 2026

The user requested a second review including optimized low-precision Muon,
the 3090 Ti compute unlock, both MiniCPM5 and Spark, and parallel multimodal /
backbone work across the two GPU pairs. The
[follow-up report](docs/experiments/20261026_Mixed_Precision_Muon_and_Parallel_GPU_Budget_Review.md)
corrects the missing unlock: primary-source 3090 Ti BF16 GEMM is 1.80× faster
with FP32 accumulation. TITAN RTX already has full-rate FP16 accumulation and
does not receive another 2× factor or a native-BF16 claim.

At 1B total processed positions, the unlocked pair's conditional full-schedule
budget is 353–1,325 GPU-h for the current 2.5168B MiniCPM backbone, 424–1,590
for a 2.8B complete-module MiniCPM envelope, and 627–2,349 for the full 4.4B
Spark envelope. CUDA phase delivery remains assumed, with slower fallbacks
reported. Full Spark's H200 Muon budget is 138–501 GPU-h per schedule or
421–1,518 for three equivalents plus qualification; the fused-AdamW control
is 118–429 per schedule. The M2 BF16 projection remains unqualified with a
null executable budget, despite potentially lower persistent Muon state.

The review distinguishes overlapping independent warmups from a connected
joint autograd pipeline. Moving live encoders/bridges to TITAN can relieve
3090 residency but does not halve a backbone-bound schedule. All original
parameters and live joint-stage sensory paths remain in scope. The calculator
and evidence were committed at `25aa426`; final report validation checks the
pipeline placement against the pinned component ledger as well. No training,
driver/firmware modification, paid provisioning or change to optimizer defaults
and resource guards was performed.

## Spark formal full-training compute review — 26 October 2026

The user requested total compute estimates for the modified SparkX 2.6 4B
architecture on DigitalOcean, this M2 Ultra, and independent dual-3090-Ti and
dual-TITAN jobs. The user selected 100M/1B/5B total processed-position scenarios.
The [review](docs/experiments/20261026_SparkX_Formal_Full_Finetuning_Compute_Estimate.md)
preserves full joint trainable scope and the current native-fused optimizer
policy. It separates the 4.3208B declared floor from a rounded 4.4B planning
placeholder, measured MiniCPM timing from assumed CUDA delivery, numerical
precision from trainable scope, and occupied GPU-hours from wall time.

At 1B positions, a mixed-precision H200 schedule is conditionally estimated at
116–417 GPU-hours; three schedule equivalents plus initial qualification are
355–1,266 GPU-hours, about $1,590–$5,660 at the checked $4.47/GPU-hour rate.
Strict FP32 H200 sensitivity is 495–1,484 GPU-hours for one schedule. Each
dual-24-GB pair needs qualified sharding/offload. The full FP32 Mac state floor
exceeds both physical memory and the unchanged 50 GiB driver guard; its hour
projection is explicitly compute-only, not an executable configuration.

The [calculator](scripts/estimate_sparkx_training_budget.py),
[assumptions](configs/sparkx_formal_training_budget_estimate_v1.json) and
[evidence package](reports/sparkx_gpu_hour_estimate_20261026/README.md) preserve
15 hashed input snapshots, the newer 12-update MiniCPM sustained evidence and
current official hardware/pricing references. Nine offline checks pass,
including deterministic fresh-CLI replay, original phase/changed-tensor audits,
Decimal hour/cost arithmetic and separate-pair GPU occupancy. No new model/GPU
job, dataset admission, paid resource, guard or training-default change occurred.
The next measurement is a separately admitted bounded Spark backend/residency
and representative multimodal/learning-graph pilot; final data volume, extension
inventory, numerical qualification and product convergence remain unestablished.


## Integrated repairs and diagnostics — 26 October 2026

**Current work — memory v2, vision v2, fused continuation instrumentation and
offline Spark preparation are committed. Mechanical acceptance is separate
from learned quality. Memory and vision quality remain rejected; exact native
fused continuation, external asset admission and native Spark remain open.**
The protected untracked `src/replica_lab/runtime_pilot.py` is unchanged.

Latest focused commits: `edcf1d47` memory refresh, `73f6cb64` caption target
identity, `c5e57b01` fused boundary instrumentation and `5c6577e9` Spark
residency/tranche replay. All native jobs used the sealed MiniCPM5 source,
MPS with CPU fallback disabled, one exclusive worker and the declared
50 GiB driver / 512 MiB swap-growth / 15% host-free policy.

### Memory v2 native result

`native02` and `restart02` under `runs/memory_causal_fresh_20261026/` completed
the prepared04 v2 protocol. The restart encoded 216 new queries, re-encoded
zero old observations/queries and passed byte-identical same-query parity.
There were 72 observation commits, zero durable refresh commits and 1,056
query-local applied refreshes; all 144 shuffled pairs perturbed state/value
outputs while durable state stayed unchanged. Mechanical gates, checkpoint
bindings, revocation/stale enforcement and resource limits pass.

The immutable restart rankings replay through `memory.acceptance.v3` into
`reports/memory_acceptance_20261026/causal_fresh_restart02_acceptance_v3.json`
and `receipt_manifest_v3.native02.json`. All 17,520 rows validate. Candidate
quality is rejected: 21/384 accepted-correct, 15.22% accepted precision and
30.26% coverage; restart-held-out is 2/64 accepted-correct at 15.38% precision.
The full-cohort oracle diagnostic is 384/456 raw top-1 (84.21%), below its 0.85
bound, so the native disposition remains `inconclusive`, not accepted. No
learned selector or useful-memory claim is made. Deterministic checkpoint
crash-injection recovery passes; no SIGKILL/native crash test is claimed.
See [causal results](docs/experiments/20261026_Causal_Fresh_Selectors.md).

### Vision v2 native result

Train02 exposed the donor/target case-binding defect; `73f6cb64` fixes it and
adds a runner regression. Train03 was only an invocation failure from omitted
fallback-disabled environment. Train04 then completed 24 projector-only
updates with six trainable tensors and finite nonzero backward-only gradients.
The scorer positive/negative contracts pass, but train-fit answers are 3/14
and dev answers 2/18; caption semantic F1 and following accuracy are zero.
The run stopped at `stopped_before_audit`; held-out was never read and reload
was not authorized. Retention stayed 22/32. Raw outputs and the unpromoted
diagnostic extension remain under
`runs/vision_balanced_alignment_v2_20261026/train04/`.
See [vision results](docs/experiments/20261026_Vision_Balanced_Alignment.md).

### Fused continuation and Spark preparation

The fused boundary instrumentation passes 33 training-state/fused tests and 25
optimizer tests. A/A variation is reported separately from A/B exact failure;
historical comparison09/10/11 failures and resource stops remain unchanged.
No new native continuation or replacement run was authorized.

Spark residency v2 explicitly separates 12-, 14-, 16- and 18-byte plans, the
mixed FP8/BF16 source inventory and unqualified backend/dequantization gates.
The calculator and strict receipt check pass. The 32-input research bundle
verifies with Git tracking enforced and no network or payload copy. Only six
owned synthetic protocol fixtures are admitted; external media, rights
clearance, asset QA and native Spark are not completed.
See [residency](docs/experiments/20261026_Spark_Residency_Reconciliation.md)
and [dataset preparation](docs/experiments/20261026_Spark_Dataset_Preparation.md).

### Verification and next bounded work

| Verification | Result |
|---|---:|
| Memory fresh + v3 + v2 acceptance suites | 33 passed |
| Visual caption + balanced alignment suites | 27 passed |
| Training-state + fused continuation suites | 33 passed |
| Optimizer suite | 25 passed |
| Spark residency + first-tranche suites | 35 passed |
| Spark residency receipt / strict bundle | both verified |
| Full repository regression | 2016 passed, 1 skipped, 21 subtests |

The next justified work is a predeclared visual learnability hypothesis only
if the failed fit/dev analysis supports it, and explicitly authorized external
asset/right clearance before any real tranche. Do not open the sealed visual
audit after the failed gates, do not tune against it, and do not infer native
Spark or exact fused-resume success from software receipts. The final full
suite, including checkpoint crash injection, passed in 292.73 s with one
explicit skip and one warning. Raw evidence is
`reports/regression_20261026/full_suite.log` (SHA-256
`89005405637e13a0ba33dc316fbdc708d1d92d0d5ed152c2378343112a686f23`)
and `full_suite.xml` (SHA-256
`1564943f57a6d88fb1379ad1c7d1e68c71dd042a7b584db6ce0aef4459e458ea`).


## Full-scope training performance investigation — 25 October 2026

The user requested training-code analysis, improvements on M2 Ultra without
reducing trainable scope, and a market-GPU comparison. Cached phase markers show
17.51 s/update in the post-backward gradient-check/clipping/optimizer phase of
the 55-token full run, versus 2.20 s/update forward/backward. The five-step FP32
Muon matrix updates account analytically for 97.65 TFLOP per optimizer update.

The [bounded protocol](configs/full_training_throughput_probe_v1.json) declares
three updates per arm on the existing eight-row diagnostic. The new profiling
entry point verifies all 2,516,756,480 original parameters /381 tensors are
trainable, optimizer-owned, receive nonzero gradients and actually change.
Reference Muon and fused AdamW, batching and SDPA are separate declared arms.
The unchanged 50 GiB driver /512 MiB swap /15% free guard and exclusive MPS
worker apply. No weights are exported or selected and no original is downloaded.
Optimizer substitutions are throughput comparisons, not convergence equivalence.

Pre-run software validation: `tests/test_full_training_profile.py`,
`tests/test_optimizers.py` and `tests/test_capacity_guard.py`: **40 passed in
2.76 s**. Native outcomes and the requested reports follow separately. Current
vision-curriculum and other concurrent changes are outside this package.

The protocol/harness was committed at `f25ea86`. Fresh reference Muon and fused
AdamW both completed three updates with all 381 original tensors changed.
The matched probe rates are 0.04344 versus 0.11175 updates/s (about 2.57×).
Both passed the unchanged guard. The combined gradient-check/clipping phase
takes about 5.5 seconds; an additive timing-only refinement separates its scope
audit from norm clipping in the remaining predeclared batching/attention arms.
It changes neither the updates nor the admission/resource policy. Initial
worker-lock refusals are retained and caused no model load.

The completed [performance report](docs/experiments/20261025_M2_Ultra_Full_Training_Performance.md)
and [market GPU comparison](docs/research/m2_ultra_training_performance_20261025/GPU_Comparison.md)
include the [native evidence summary](reports/m2_ultra_training_performance_20261025/native_summary.json).
Mean update times are 23.021 s for Muon and 8.949 s for fused AdamW; sampled
driver peaks are 30.08 and 40.03 GiB. All initial tensor hashes, tokenized data
and the first loss match, and both arms change all 381 tensors. The 5.5 s phase
combines exhaustive gradient auditing with clipping; the timing split has not
run natively. Batch-4/SDPA arms are deferred because the shared worker was busy.
The production trainer's AdamW option remains custom eager AdamW; the measured
native fused path is profiling-only. No default or checkpoint is promoted.
Persistent FP32 state is at least 24.586 GiB for current Muon and 37.503 GiB
for all-AdamW, before activations/temporaries. Exact CUDA equivalence remains
unmeasured. The timing refinement's two scope/collation tests also pass in 1.18 s.
Final offline [evidence validation](reports/m2_ultra_training_performance_20261025/validation.json)
reproduces the cached analysis and verifies native provenance, matched initial
conditions, six updates, all changed tensors, resource limits and report links.

**Current implementation — staged curriculum rejected on learned quality; write-position interference isolated in memory; balanced image dependence supported on procedural holdouts.**
Caller-bound canonical context verification, exact seeded schedule audits and
versioned representation-preserving rebuild pass 140 integrated tests and a fresh
process replay smoke. The tracked source export passes **1,372 tests +21 subtests
in 166.68 s**. No native memory rerun occurred; the v5 candidate remains rejected.
See [repair qualification](reports/memory_contract_repairs_20261025/qualification.json).

## Verification repairs — 25 October 2026

The [verification repair report](docs/experiments/20261025_Verification_Repairs.md)
closes all four reported defects. Raw pages now require explicit record
dependencies or fail closed; omitted provenance cannot be rebound or saved.
Contained-record revocation is rejected before and after reload. Save/reload
parity pairs every `(world, fact, layer)` and rejects duplicate/missing rows;
one-/two-/three-fact immutable replays still pass with 16/16/24 complete parity
rows per path and zero difference. The old frozen-feature verdict replays as
`inconclusive_visual_contribution`, not projector-bottleneck proof. Bounded-RSI
unit tests use synthetic fixtures; only explicitly marked historical integration
evidence requires provisioning.

The complete verification-repair set passes **147 tests with two explicit
integration skips in 35.86 s** from a clean `git archive` export. The bounded-RSI
set alone passes 11 with one skip and 12/12 when provisioned. These are focused
clean-export results, not capability claims or an artifact-free full-suite claim.
The next native package is the fresh paired balanced image-dependence diagnostic;
matched-episode memory localization follows.

## Balanced image-dependence diagnostic — supported on procedural holdouts

The [balanced diagnostic results](docs/experiments/20261025_Vision_Dependence_Results.md)
use 180 worlds /360 paired images with split-specific target and distractor pools,
same-question shuffling and world-clustered uncertainty. The native MPS run and
one-shot finalizer complete with zero updates; independent review finds no
material discrepancies.

| Split | Primary | Question-only | Shuffled primary | Gain over question-only [95% CI] | Gain over shuffled [95% CI] |
|---|---:|---:|---:|---:|---:|
| Dev | 71/120 = 59.17% | 36/120 = 30.00% | 50/120 = 41.67% | 29.17 pp [21.67, 36.67] | 17.50 pp [8.33, 25.83] |
| Final | 75/120 = 62.50% | 36/120 = 30.00% | 46/120 = 38.33% | 32.50 pp [25.83, 38.35] | 24.17 pp [14.17, 35.00] |

The aggregate gate passes, led by spatial accuracy 24/24 on both splits. It does
not establish per-task competence or real-image perception. Count-total cues
remain on 12/72 rows, pair-twin donors occur, and lexical diversity is limited.
Peak sampled driver is 11,815,763,968 bytes with zero swap growth and 76%
minimum free. The full suite passes **1,706 tests +21 subtests in 207.54 s**.
Next run matched-episode memory localization before any new projector arm.

## Matched-episode memory localization — exact-key retention fails first

The [matched-memory results](docs/experiments/20261025_Matched_Memory_Localization.md)
use identical episodes across exact observed-key retrieval, query addressing,
candidate/value discrimination and native generation. The first failing stage is
exact-key stored-value retention/readout: top1 **18/120**, mean rank 3.275,
margin −0.0219221. Query-derived addressing passes **116/120**, mean rank 1.0333,
but does not establish stored-value recovery. Candidate/value discrimination
fails with the same 18/120 top1.

The failure is last-write dominated: **18/18 final-write targets top1 versus
0/102 non-final-write targets**. Corrected targets are 12/12; unaffected retained
facts are 0/30. Native03 audits and applies a 12-token answer budget, removing
native02's truncation confound: exact source is 162/162, full context 160/162,
memory 0/162 and context-removed 42/162. Source-grounded generation works;
learned-memory generation does not. Save/reload is 42/42 equal with zero updates.
Guard elapsed is 543.251 s, peak driver 10,765,336,576 bytes, zero swap growth
and 74% minimum free. Independent CPU recomputation and outcome review confirm
the localization. Next compare update dynamics on the same episodes before any
writer/reader change or new training arm; the staged vision curriculum follows.

## Update dynamics — write-position interference isolated

The [update-dynamics results](docs/experiments/20261025_Memory_Update_Dynamics.md)
compare 20 frozen variants on identical prepared02 observations. Per-depth and
shared-mean full baselines both recover only 18/120 targets, so shared reduction
is not the first cause. Reversing writes reaches 42/120; prefix-through-target
and target-only recover **120/120 in both reducers**. Refreshing the target last
reaches 120/120 per-depth and 119.5/120 shared. No forced-full gate variant
approaches the exact recovery gate. Query-derived addressing stays invariant at
232/240. This isolates write-position/recency interference and justifies only an
order-aware/refresh diagnostic with the writer frozen, not writer training or
per-depth promotion. Independent review confirms all 22,224 rows, 1,464 ordered
histories and candidate inventories. Guard elapsed is 86.923 s, peak driver
11,815,763,968 bytes, zero swap growth and 78% minimum free.

## Staged vision curriculum — engineering pass, learned quality rejected

The [staged curriculum results](docs/experiments/20261025_Staged_Vision_Curriculum_Results.md)
complete Muon-8bit and AdamW arms with identical fresh initialization, 128 updates
/512 examples and 32 probe updates /64 examples each. AdamW is selected by mean dev
loss 0.797576 versus Muon 6.707518 and finalizes once. Dev correct is **99/240**
versus no-image 65/240, wrong 92/240 and shuffled 87/240. Audit correct is
**37/120** versus no-image 29/120, wrong 38/120 and shuffled 32/120. Every
accuracy/control margin and abstention gate fails. External final captions have
token-F1 0 on all 8. Text retention is unchanged at 22/32, with component
nonregression passing but baseline ineligible. Engineering, strict export,
zero-update fresh-process reload and independent scoring pass; the curriculum is
rejected on learned quality. The [supplemental lineage](reports/vision_curriculum_20261025/supplemental_lineage.json)
binds outer guard, reloads, reservations and the seal-repeat refusal. Next analyze
constant-output priors and abstention before any larger training arm; do not
enlarge the connector or claim optimizer causality.
The final full suite passes **1,863 tests +21 subtests in 228.22 s**.

## Verified learning and visual-memory preparation — contracts complete, evidence gated

The transaction and payload-binding work is complete: proposal/world/code/identity
authentication, coordinated publish/rollback, durable recovery and CLI recovery
all pass focused tests and an actual procedure cycle. Verified-learning contracts
separate demonstration recall from unseen rule transfer, matched feed-forward
and recurrence controls, frozen-vs-updated policy accounting and personality
fidelity. The CPU fixture reports demonstration 1.0 but transfer 0.0625 and
inconclusive policy comparison; it is not verified-learning or RSI evidence.
Full memory/vocabulary/learning factorial work remains deferred.

The delayed visual-memory corpus has 12 episodes, 24 images and 96 queries with
object/region identity, coordinates, timestamps and provenance. CPU contracts
pass: memory and exact/full-context each 64/64, shuffled 20/64, zero/disabled/
context-removed 16/64 (expected-unknown abstentions). Native visual-memory
evaluation remains blocked: V1 perception is rejected and matched memory-use
fails exact-key retention. Balanced frozen-feature information does not remove
that gate. Verified-learning evidence and visual-memory quality remain unproven.

## Isolated V0 source and native preregistration — 25 October 2026

The production adapter/session/frame path now supports one source-owned image,
196 frozen spatial patches and a new width-2048 projector. Response-only labels,
input gradients through the frozen decoder, task isolation, producer/cache
invalidation and strict extension reload pass **82 focused tests in 3.91 s**.
The runner helpers also execute real tiny-decoder backward and Muon updates on
CPU fixtures; these are software checks, not pretrained vision qualification.

An actual [CPU admission](reports/vision_v0_20261025/cpu_admission.json) ran the
official processor on a pinned owned image and loaded the native tokenizer.
Selected encoder: SigLIP2 base 224/16 revision
`75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`, Apache-2.0. Acquired checkpoint
source is 1,500,804,926 bytes including unused text-tower tensors; only its
92,884,224-parameter vision component will execute. Processor interpolation is
official PIL code 2 (bilinear), not bicubic. Both explicit metadata corrections
preserve their parent manifests and change no source payload bytes.

[Initial V0 protocol](configs/minicpm5_vision_v0_qualification_v1.json):
`781f5eef48a239bc9aaed69c3b3515894fb326ac715f460379fe85a241ca8ac7`.
It pins 83 implementation/environment files, sealed baseline/source/data hashes,
six projector tensors / 5,772,800 parameters and **zero optimizer updates**.
At this initial admission, visual generation/backward and fresh-process reload were unrun.
No V1 training rows are admitted. The Pillow-only overlay preserves `.venv`;
the native guard remains 50 GiB, fallback=0, one worker, 512 MiB sustained swap
growth and 15% host-free over three samples. The v5 candidate remains rejected.

The first full tracked V0 export (`5a537df`) ran **1,523 passing tests +21 passing
subtests and one failing structure-only test in 167.18 s**. The failure reads an
ignored machine-local E1/E2 receipt. Its raw log and JUnit remain in the
[failed-export record](reports/vision_v0_20261025/clean_export_attempt01_failed.json).
That incidental test is removed, not re-pinned; six actual migration/strictness
tests pass in 2.28 s. All 83 runtime/environment implementation pins are unchanged.
Fresh tracked export `baf3347` then passes **88 tests in 4.19 s**, the full
visual-data audit and sealed-source offline reuse. The unchanged prior passing
checks are reused, not presented as a second full-suite execution. The
[source admission](reports/vision_v0_20261025/source_qualification.json) permits
only bounded V0 inference/backward/restart under the preregistered protocol.

## Native V0 attempt01 and targeted constructor repair

Attempt01 failed before encoder inference because the production service omitted
its optional `_preprocessor` initialization; the tiny fixture factory set it.
The guard itself did not stop: **11,815,763,968-byte peak driver**, zero swap
growth, 76% minimum host-free and 8.822 s guard elapsed. The actual native
text-only bypass logits are bitwise equal (`[1,22,130560]`, maximum difference 0).
No visual generation, projector backward or optimizer update completed.
The [failed disposition](reports/vision_v0_20261025/native_attempt01.disposition.json),
child log, guard, raw failure and profiler records remain preserved.

One targeted fix initializes that hook to `None` in the real constructor.
Twenty-four feature/runner tests pass in 7.95 s. Corrected
[protocol v2](configs/minicpm5_vision_v0_qualification_v2.json)
(`a1fdfb12ad204011de5761348c03e1f555ca4676ee660b5866a8dd840211ef36`)
changes only this code pin; data, seeds, cases, gates, limits and zero-update scope
remain fixed. Native passing-after verification and fresh-process reload are
still required; V1 remains unadmitted.

Before retrying, a new consumer regression reproduced stale cache acceptance
after an encoder position buffer changed. Live producer stamps now cover both
parameters and buffers. The failing-before XML and the subsequent wording-only
test failure are retained; the obsolete message assertion was deleted, not
re-pinned. **25 feature/runner tests pass in 2.45 s**, including buffer mutation
rejection and cache eviction.

[Protocol v3](configs/minicpm5_vision_v0_qualification_v3.json)
(`7d0089e999ce505a4331f6ffbb1581846f13e9a20b46e0199734a5e46bc0b36a`)
contains these live-producer guards. V2 was committed but never run. V3 retains
the original data, seed, cases, limits, numerical gates and zero-update scope.

## Native V0 attempt02 and fresh-process restart01

Both complete under protocol v3 with fallback disabled. Six FP32 projector
tensors receive finite gradients through the frozen decoder; base and encoder
gradients are absent. No optimizer is instantiated and no tensor is updated.
All runner checks pass. Correct/wrong/no-image generation produces 24 tokens.
First-token logit contrasts are 5.459615 and 16.943596; these prove influence,
not perception. The exported v4 extension has 49 tensors and no base copy.

The fresh process loads that exact extension with zero updates and reproduces
all three token sequences and logits bitwise (maximum difference 0). Primary
and restart profile times are 9.934 s and 8.379 s; both sampled driver peaks are
11,815,763,968 bytes. The primary guard records zero swap growth and 76% minimum
host-free. These are mechanical interface results, not useful visual grounding.

The initial CPU score deferred only for missing restart. After restart, the
original scorer failed with `KeyError: rows`: its consumer expected raw rows
from a summary. A scorer-only repair now reads the already-authenticated report
rows and publishes an actual report-file locator. A strict replay cannot change
native code, data or gates; it pins both native report hashes and changes only
the independent scorer. Its **22 provenance/audit tests pass in 2.18 s**.

[Scoring replay](configs/minicpm5_vision_v0_scoring_replay_v1.json)
(`23f696b3fff05d8e7b4d02a44ecee4145ebb9b2430a6b4bb27618af342ada25e`)
is retrospective to the completed native executions and committed before CPU
replay, not a new pre-run native protocol. The failed scorer attempt is kept.
The committed CPU replay independently accepts V0 (`accepted=true`,
`engineering_pass=true`, `quality_pass=null`). The actual V1 prerequisite consumer
also accepts its immutable native report-file locator. See
[V0 qualification](reports/vision_v0_20261025/qualification.json). No perception
quality, useful learned memory or integrated RSI claim follows.

## Named projector-only V1 admission and preregistration

[V1 protocol](configs/minicpm5_vision_v1_projector_alignment_v1.json)
(`044cebb9b1eafe8a529eb42b04e98217a4b8b755fba4d465ba815ebec410a401`)
declares one 128-update arm with four accumulated examples/update, 512 exact
admitted rows, 256 verified image assets and 64 complete training worlds.
Evaluation uses 64 cases from 32 disjoint dev worlds; all come from one owned
procedural source family, not 32 independent external collections. Thirty-two
protected text cases cover seven domains. External dataset assets acquired: zero.
The eligible source inventory is 2,048 rows; evaluation rows are not training.

Six FP32 projector tensors / 5,772,800 parameters are the only trainable scope.
The two matrices use Muon int8 momentum, other tensors FP32 AdamW; weights and
gradients remain FP32. Learning rate 1e-4, momentum .95, NS steps 5, block 2048,
decay .01, accumulated norm clip 1.0, one seeded epoch and no adapters/LoRA.
The encoder, sealed decoder, memory, core and identity remain frozen/disabled.

The actual [CPU admission](reports/vision_v1_20261025/cpu_admission.json) verifies
85 source/environment pins, accepted V0/checkpoint lineage, exact schedule and
all token budgets: maximum training text+response 36, eval prompt 99, retention
prompt 42 and response including EOS 2. Dataset logical bytes are 4,475,237,
below the separate 25 GiB dataset ceiling; native RAM guard is unchanged.

The new v2 admission separates eligible rows from admitted training rows, rejects
rehashed wrong labels/questions, binds the named run and V0 producer/checkpoint,
and keeps protected splits outside training. Text retention is measured on the
initial checkpoint before updates and independently compared after; no guessed
baseline accuracy is pinned. Correct/wrong/shuffled/no-image/caption-only/OCR-only
controls share the same question template. Caption-only is an explicit oracle
scene-description upper bound; OCR control is verified absence of rendered text.

A real tiny-decoder six-control regression reproduced and fixed the no-image
auditor lookup; failing/passing XML is retained. The integrated run reports
118 passing checks and four failures in independent diagnostic slices; those
four targeted mechanical failures then pass after correction. Unchanged passes
are reused rather than described as a second suite execution. V1 native training
is not yet run. V2 and learned V3/RSI remain gated, not implicitly admitted.

Clean tracked export `2dcc705` passes **1,541 tests +21 subtests in 176.69 s**
with one existing sparse-invariant warning. From that export, all 85 source pins,
the real accepted V0/checkpoint dependency, 512 admitted rows, exact 128-update
schedule and scorer imports validate. Sealed model/parent runtime inputs are
explicit APFS clones, not tracked payloads. See
[V1 source admission](reports/vision_v1_20261025/source_admission.json).

V1 attempt01 then fails before model allocation: the runner reads `split` from
the intentionally label-free model rows, rather than verified case metadata.
Zero updates; peak driver is 393,216 bytes, zero swap growth, 94% minimum free.
The raw guard/failure/profile and [disposition](reports/vision_v1_20261025/native_attempt01.disposition.json)
remain preserved. The host lookup is corrected; seven real tiny/admission tests
pass in 8.57 s. [Protocol v2](configs/minicpm5_vision_v1_projector_alignment_v2.json)
(`980454c29f31f1196af71889aecf4fa6a5e99da996d4351d7798c73d34ca36c2`)
changes only the runner code pin. Data, seed, checkpoint, schedule, optimizer,
resource policy and every numerical quality gate remain unchanged.

## Measured projector-only V1 disposition — rejected

V1 attempt02 completes all **128 updates /512 actual backward calls** under the
unchanged preregistered gates. Exactly six projector tensors change; 43 other
extension tensors and the original backbone remain unchanged. The encoder is
frozen with no gradients. Independent scoring passes engineering and rejects
quality; see [V1 qualification](reports/vision_v1_20261025/qualification.json).

| Control | Correct /64 |
|---|---:|
| Correct image | 18 |
| Wrong image | 20 |
| Shuffled image | 15 |
| No image | 22 |
| Oracle caption-only | 60 |
| Verified empty OCR text | 15 |

All 32 dev world groups belong to one owned renderer family. Correct-image
accuracy is 28.125%, below the 65% floor and every declared control margin.
Text retention remains bitwise unchanged at **22/32 before and after**, but
fails the absolute 75% floor. Mean training loss falls from 10.4217 to 1.7386;
that does not establish visual grounding. V2 is not admitted; no LoRA is trained.
Learned V3 and neural RSI remain blocked by perception and rejected v5 memory.

Training/evaluation take 267.930 /130.178 s; total sampled profile 416.450 s,
guard 419.767 s. Peak driver is **11,989,827,584 bytes**, zero swap growth and
74% minimum host-free. Actual optimizer state is 5,823,488 bytes: 5,767,168 int8
momentum bytes plus 56,320 FP32 state bytes; weights, gradients and Muon
temporaries are FP32. Final extension hash:
`bdc367424b2afcb2e88dba04b6b29495a3e440b5d850f8ae5bc55dbfa428c3a4`.
All raw answers, example/update traces, checks, profiles and rejection are retained.

## Second V1 wide-connector arm — engineering pass, quality rejected

The [wide V1 results](docs/experiments/20261025_MiniCPM5_Vision_V1_Wide_Results.md)
record the final attempt07/restart07/score08 evidence. The preregistered arm uses
a 4096-wide, 11,542,016-parameter projector with fresh seed-17 initialization,
the same admitted data, 128 updates/512 backward calls and unchanged quality
gates. Independent scoring passes engineering, including mandatory fresh-process
reload, but returns `accepted=false` on learned quality.

| Control | Correct /64 |
|---|---:|
| Correct image | 21 |
| Wrong image | 12 |
| Shuffled image | 15 |
| No image | 22 |
| Oracle caption-only | 60 |
| Verified empty OCR text | 15 |

Correct-image accuracy is 32.8125%, below the 65% floor and the no-image control.
Margins over wrong/shuffled are only 14.0625/9.375 percentage points, below the
20-point gates. Text retention remains bitwise unchanged at 22/32 before/after,
below the 75% absolute floor. Training loss falls 9.3908→1.7836 without
establishing grounded generalization. V2/LoRA remains unadmitted.

Exactly six projector tensors change; 43 other extension tensors, the encoder
and sealed backbone remain unchanged. A fresh process performs zero updates and
reproduces evaluation and retention JSONL bitwise. The scorer correction uses a
declared scorer-only replay pinned to native commit `ddc6a07` and both native
reports; no native gate or runner bytes change. Primary/reload peak sampled
driver is 11,956,273,152/10,768,056,320 bytes, with zero swap growth and 74/76%
minimum host free. Guard elapsed is 420.248/144.060 s. Each extension is
65,688,472 bytes; no base copy is saved.
The final full suite passes **1,615 tests +21 subtests in 188.62 s**, and the
[independent review](reports/vision_v1_wide_20261025/independent_review.md)
finds no integrity discrepancies.

Tiny-cohort diagnostics show why capacity alone is not a promotion: the
128-update wide arm fits 8/8 training worlds but scores 0/8 on fresh dev
correct, wrong and no-image controls. Attempts 01–06, restart06 and score07 are
preserved engineering failures/superseded attempts, not final evidence. The next
boundary is fresh-world error/readout diagnosis, not repeated dev-set retuning.

## Independent exact-source and bounded-host controls

The [exact visual-source baseline](reports/exact_visual_source_20261025/qualification.json)
passes 10 tests and an actual CPU CLI: 36 sources /12 worlds /one owned renderer,
36 delayed exact retrievals, restart without re-encoding, an unaffected reference
and two-record ancestor/descendant revocation. Corrections cover metadata/crop/
owner lineage only; factual memory-only recall is explicitly not evaluated.

The [bounded sandbox procedure cycle](reports/bounded_rsi_20261025/qualification.json)
passes 11 tests and actual practice/independent verification/rejection/publication/
authenticated rollback with deterministic replay. Source cases improve 0/2→2/2,
independent verification is 2/2, the rejected family is 0/1, and the candidate
artifact is revoked when the protected ancestor is restored. It performs
22 tool calls /15 evaluator runs, zero neural updates and no selector-policy
change. This is procedure plumbing, not integrated or accelerating RSI.

Fresh storage/addressing/readout diagnostics now preserve all before/after-load
raw evaluations, score only unique after-load cases, retain superseded observed
values as evaluation-only negatives, enforce exact workload limits and
authenticate target roles from source codebooks. The 29 probe/guard checks pass.
One actual tiny CPU smoke is mechanical only. The next native package is a
zero-weight-update probe of the rejected writer, not another blind training arm.

The [prepared native probe](data/memory_storage_probe_20261025/prepared01/protocol.json)
pins code `f621384`, fresh data
`579917456164252b203315005f40421f47e207b55f747a4ea373f99828dbc3e1`,
and protocol
`f4dd63df2b7abe06e2f91777c1420787d8f9aac7f64543f483b73e007bc5c83c`.
Four worlds contain three primary facts plus two independent-world distractors
each. The exact bound is 552 evaluated rows /276 unique after-load cases.
The original rejected v5 writer hash `ff339867...c11f49` stays frozen.
[Execution admission](reports/memory_storage_probe_20261025/native_admission.json)
declares zero updates and the unchanged exclusive fallback-off 50 GiB guard.
No stored-observation vector recovery is useful language memory by itself.


## Integrity, reproducibility and new diagnostic evidence — 25 October 2026

**Persistent rebinding repaired.** `bind_source()` now proves consumed lineage
against the exact authoritative view: record-backed events must match an eligible
record byte-for-byte or be superseded originals whose eligible correction was
consumed later; unbacked observations must not repeat revoked source text; every
eligible correction in the view must already be reflected. Reference-bound load
re-runs the same check. Revoke→rebind is refused for both representations;
verified current-view rebuild restores eligible lineage and persistence. New
`tests/test_persistent_rebinding.py` covers direct rebind, append-clear,
correction/supersession, unaffected records, scope changes, offline inspection,
load-then-rebind and failed rebuilds. Full suite: **1,587 tests +21 subtests**
(**1,604** after one-write/interference additions).

**Reproducibility.** V0, V1 and the memory-diagnostic verdicts were reproduced
from pinned-commit worktrees with cloned ignored inputs: V0 accepted identically,
V1 rejected identically, memory rescore matched identically. Only absolute
worktree paths and timestamps differ (`reports/reproducibility_replay_20261025.json`).
Artifact-dependent tests now declare inputs with provisioning commands instead of
silent skips (`tests/conftest.py`, docs/maintenance/20261025_Test_Provisioning.md).
Spark offline preparation now retains the pinned tokenizer/vocab/merges payloads
(29,432,928 bytes) in `data/cache/spark_sources`, with a header-bundle writer and
byte-identical offline reproduction; 26 unit tests pass.

**One-write storage discriminator (native, zero updates).** Eight single-fact
worlds, exact observed key, per-depth/shared/cross/zero read paths, before/after
save-reload. Per-depth mean cosine **0.929**, shared **0.923**, zero exactly 0,
save/reload bitwise exact. Independent CPU rescore matched with 8 authenticated
state images. Two-fact and three-fact interference runs also recover
(**0.877** and **0.853**). **Earlier multi-fact rejection is not a basic write
failure**: storage works incrementally; addressing/selection and correction
semantics are the remaining failure class, not a missing write path.

**Vision information probe (native, zero updates).** 96 frozen-encoder cases
(24 train + 24 dev worlds). Combined frozen-feature linear readout reaches
0.4167 dev accuracy (chance 0.0833, majority 0.1667), question-only 0.4375,
image-only 0.1875, shuffled-image 0.375. Repaired decision replay now returns
`inconclusive_visual_contribution`: combined features underperform question-only
and do not show the declared incremental gain. The earlier
`frozen_features_support_readout` interpretation and projector-bottleneck claim
are superseded; raw measurements remain unchanged. See
[conservative replay](reports/vision_diagnostic_20261025/conservative_replay.json).

**Tiny-set overfit diagnostic (projector-only).** At 16 updates and at 32 updates
(two/four epochs over eight train worlds), train-fit is **0/8 on every control**,
and fresh dev is also 0/8. This is a training/supervision failure, not useful
overfitting. Generated text is garbage tokens, not answers. Next discriminator:
`scripts/analyze_overfit_fit.py` compares teacher-forced response loss and
response-token argmax with greedy decoding to separate learning failure from
decoding failure. V2/LoRA remains unadmitted.

**Retention protocol.** `component_nonregression.v1` separates baseline
capability eligibility (`baseline_capability_ineligible`) from component
nonregression (drop and token parity); legacy combined gate is unchanged for
historical protocols. The frozen 22/32 baseline stays ineligible for the
absolute 75% floor; the completed V1 experiment is not rewritten.

**External datasets.** The earlier eight-entry audit reserved four evaluation
sources and blocked GQA/VSR/LLaVA-Instruct on pins or the COCO family. A bounded
LLaVA-Pretrain tranche now admits **30 inspected root assets /30 normalized
rows**: 15 train, 7 development and 8 sealed untouched evaluation. Acquisition is
61,174,504 bytes; every admitted image is decoded and digest-verified, split and
label/input leakage checks pass, and rebuild is deterministic. See
[intake report](reports/external_caption_tranche_20261025/intake_report_v2.json).
Use is internal research only; images are not redistributable. Caption origin is
recorded as ambiguous between raw CC-3M and identical BLIP text, and no semantic
caption–image agreement screen is claimed. The earlier audit remains preserved.

**Spark backend.** transformers 5.20.2 lacks `spark2_5`, compressed-tensors is
absent, and MPS cannot allocate float8_e4m3fn tensors. FP8 storage/compute is
therefore unqualified here; silent dequantization would not count as FP8
qualification (`reports/spark_qualification_20261025/backend_audit.json`).
Dedicated `Spark2_5` preparation contracts now validate official geometry,
layer schedules, tokenizer/template identity, extension-only inventories and
explicit BF16/native-FP8/dequantized-derivative identities; 17 focused tests
pass, and existing Spark preparation tests remain green. They do not execute
weights or change the baseline. Load, inference, gradients, updates and native
FP8 export/reload remain blocked by the audited backend.
## Frozen native storage diagnosis — useful storage not accepted

Native probe01 completes **552 physical evaluations /276 unique after-load
cases**, with zero updates and no changed tensors. Independent fresh CPU
rescoring matches every gate/summary and authenticates all 12 persisted images
plus before/after raw parity sets. See
[native diagnostic qualification](reports/memory_storage_probe_20261025/qualification.json).

Observed-write-key top1 is **14/56**, with only **2/56** confident recoveries.
Canonical addressing is 28/112 top1; native-reader top1 is12/56 with zero
confident recoveries. Corrected/unaffected confident recoveries are0/12 and0/24.
Storage, addressing, correction and native-readout gates fail. Source bindings,
stale-source rejection, revocation and same-process save/load parity pass.
Mechanical persistence is not useful learned memory. Copy/pointer remains
unconstructed because the storage gate fails; v5's rejection is unchanged.

Guard elapsed is80.490 s; peak driver10,763,304,960 bytes, zero swap growth and
78% minimum host-free. All encoder/model inference is native MPS, fallback=0.
There is no new memory-training arm. The whole source export at `1c609c3`
passes **1,572 tests +21 subtests in183.02 s**, with one existing sparse warning.
[Final source validation](reports/memory_storage_probe_20261025/final_source_validation.json)
records the explicit external model inputs and exact archive hash.

A separately labelled retrospective CPU geometry control leaves canonical gates
and rejection unchanged: all56 perfect observed-value vectors can satisfy the
declared margin, while centering actual observed-key readouts worsens top1
from14/56 to12/56. Thus neither label distinguishability nor a simple mean
offset explains away the failed storage result. See
[geometry analysis](reports/memory_storage_probe_20261025/observed_code_geometry.json).

**Next smallest discriminating package:** preregister one fact /one write under
the exact observed key, with full shared-depth reduction versus isolated
per-depth proposals. This distinguishes writer/gate failure from cross-depth or
multi-fact interference before any further writer, reader, pointer or LoRA tuning.
V1 is rejected, V2/learned V3/neural RSI remain unadmitted. Spark precision/backend
qualification remains separate and unrun; no baseline change or model promotion.



## Repository maintenance — 25 October 2026

The user authorized focused local commits for the accumulated work and future
completed packages. Starting HEAD: `3f1433b`, branch
`codex/d2-capacity-runtime-pilot`. Follow the
[tracking and commit policy](docs/maintenance/20261025_Repository_Hygiene.md).

The first package expands ignore rules and the index guard for local data,
payload archives, LFS object stores, caches, credentials, fixture scratch and
the protected pilot. Historical raw reports and small synthetic fixtures remain
versioned. The initial index has 3,391 files; all 5,649 stored blobs are below
50 MiB, with an 8,148,167-byte maximum. No payload untracking or history rewrite
is needed. Git LFS is unavailable; model payloads remain wholly local.

Validation: `.venv/bin/python -m pytest tests/test_git_artifacts.py -q
-p no:cacheprovider` passes **65 tests in 10.59 s**;
`python3 scripts/check_git_artifacts.py --scope tracked` and
`git diff --check` pass. Subsequent commits record stable existing evidence,
documentation and preparation packages separately. Concurrent V0 implementation
is excluded until its owner completes its qualification. No native work is
launched by this maintenance.

Stage 1 committed as `a2dc0cf`; `core.hooksPath=.githooks` is enabled locally.
Stage 2 archives 53 previously untracked historical evidence files (412,375
bytes): M0a/M2 guard traces, recovery verdicts, v5 scorer/restart traces and the
dated recovery record. All 17 JSON files, 1,410 JSONL rows and one JUnit document
parse. These are preserved past results, not experiments rerun or prospectively
registered by this maintenance; no historical payload was edited.

Stage 2 committed as `a7a2097`. A concurrent session then committed the Spark
and visual preparation package as `7418b99`; it is retained as its own package.
Independent maintenance validation ran `tests/test_spark_preparation.py` and
`tests/test_visual_data.py`: **37 passed in 0.42 s**. The visual audit-only CLI
also passes over 1,220 owned images and 2,688 candidate rows, with zero external
acquisitions and zero training admission.

Stage 3 records the accumulated architecture and Spark/Dynamic Engram research
documents, and reconciles active entry points with the existing R1–R4 repair
qualification. Both research stage graphs are acyclic; analytical costs and
historical source pins reproduce. The 51-file design package has 528 valid local
links. Recorded research and model test results retain their original dates and
scope. This stage does not qualify the concurrently edited native V0 code.

Stage 3 committed as `7854a3f`. A clean export revealed two absolute symlinks
under `reports/memory_reliability_v2_attempt03/smoke_fixture/run/` pointing to
local `runs/` payloads. Stage 4 removes only their index entries and ignores them;
both working symlinks and targets still exist. The guard now rejects staged
absolute/escaping/local-artifact symlinks. The focused guard suite passes
**74 tests in 5.86 s** and all **4,752 tracked files** pass the full-index guard.
Old commits retain only the small symlink texts; no model bytes were tracked.

Stage 4 committed as `d163909`. Its clean export passes **111 focused tests**,
but the full visual fixture audit exposed absolute checkout paths in the stored
allowlist. Stage 5 makes new locators relative and validates historical locators
across relocations without changing data, source hashes or admission fields.
`tests/test_visual_data.py` passes **19 tests in 0.50 s**, including five new
relocation/tampering regressions; the original 1,220-image audit still passes.
All existing generated assets and provenance files remain byte-identical.

Stage 5 committed as `7925cfd`. Final validation from a clean, relocated
`git archive 7925cfd` export passes **116 tests in 6.40 s** and the complete
visual-data audit. The [maintenance audit](reports/repository_hygiene_20261025/audit.json)
inventories 4,752 tracked files, 162 Markdown documents, 189 text/PDF documents,
1,150 JSON files, 53,612 JSONL rows and 577 Python files. All current-document
links, code fences and syntax checks pass; all seven R2 archive hashes match.
The eight-page architecture atlas was extracted, rendered and reviewed.

The archive's already-missing `validation_report.json` attachment is recorded
without altering historical bytes. Eleven links intentionally target ignored
local model/run evidence. No blob exceeds 50 MiB; the maximum remains 8,148,167
bytes. The protected pilot, historical evidence and all visual data bytes are
unchanged. Only the concurrent native V0 implementation remains outside this
maintenance's commits and validation. No remote push or history rewrite occurred.

## Current evidence and limits

- The committed v5 protocol trained one writer-only D1 arm for 48 updates / 192
  cases: 14 tensors and 5,251,072 parameters. Update-two gradients reach all 14
  writers; exactly 14 tensors changed and 53 remained unchanged. The original
  backbone, readers and gates stayed frozen.
- Independent routes contain 128 fast-state and 64 gate-off cases. There are 192
  actual backward calls, including 64 gate-off calls on non-fast graphs. These
  are explicit invocation records, not per-example gradient attribution.
- Candidate fast and disabled both score **56/192**, with **0/64 initial facts,
  0/16 corrections and 0/48 unaffected facts**. Their tokens differ on 128
  factual routes, so influence is nonzero but not useful. All fact floors and
  both delayed-above-controls gates fail.
- Selective exact context scores 184/192 and full context 176/192. The reference
  bundle still rejects: general accuracy is 24/32 and full context wins two RAM
  cases, violating prospective non-regression. Exact fact retrieval remains a
  separate working control, not learned memory.
- The corrected scorer decoded 1,920 rows, authenticated 32 images/source
  bindings, and accepted fresh-process mechanical restart: seven answers,
  token/state parity, seven-entry lineage and source-view equality. Restart
  persistence of wrong answers is not memory quality or power-loss recovery.
- One 48,948,272-byte extension was saved with no base copy. Training/eval02/
  restart child times are 637.228/548.855/7.182 seconds. Peak sampled MPS driver
  is 11,815,763,968 bytes, with zero swap growth and 76% minimum host-free.
- Final clean-export validation: **1,311 tests +21 subtests in 148.69 s**, and
  **49 focused reliability tests**. One existing sparse-tensor warning remains.
  See [reproducibility evidence](reports/reproducibility_v5_source_export_20261024.json).

Two scorer defects were fixed without editing native history: `eff2079` repaired
imports/gate-off accounting and `5aa7e75` canonicalized JSON audit comparison.
Eval01 is preserved as a failed scoring attempt; eval02 is the canonical
no-training replay. Total guarded wall across training, two evaluation replays
and restart is 1,751.69 seconds.

## Current blocker and next action

R1–R4 are implemented and verified; [document 17](docs/architecture/17_Memory_Reliability_Recovery.md)
records the new APIs and the historical reproductions. Next implement isolated
MiniCPM5 V0 with memory/core/identity disabled, using one sealed pretrained patch
encoder and a new projector. Native V0 is not yet qualified. The v5 rejection
satisfies package completion but not learned-memory quality. Continue fresh
storage/read separation independently; do not repeat blind writer/reader training.

Prepare official Spark FP8/BF16 sources/backend compatibility and eligible datasets
now. Native Spark work follows stable reusable interfaces and its own gates; it
need not wait for RSI success and does not block MiniCPM5. See the
[integrated design](docs/architecture/19_Integrated_Memory_Learning_and_RSI_Design.md),
[Spark plan](docs/architecture/20_Spark_X2_6_FP8_Preparation_Plan.md),
[data curriculum](docs/08_Dataset_Acquisition_and_Curriculum.md) and
[revised loop prompt](docs/architecture/16_Loop_Agent_Instructions.md).

The previous review's 49 focused tests, 27 pins and saved-answer rescoring remain
separate historical review evidence. New repair execution is the 140-test focused
set, fresh-process software smoke and 1,372-test clean export above. Spark source
and owned visual-data preparation now have executable artifacts; their native and
training qualifications remain separate.

Verify the current repair interfaces without a native rerun:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_exact_context_consumer.py tests/test_training_schedule_audit.py \
  tests/test_persistent_representation.py tests/test_exact_memory.py \
  tests/test_memory_reliability.py tests/test_fact_memory.py
```

The historical v5 scorer must run from its recorded source revision/pins; current
code intentionally does not impersonate that old protocol. No old pins were reset.

Preserve all rejected candidates, raw reports and the untracked protected
`src/replica_lab/runtime_pilot.py` unchanged.

## Documentation and source preparation — 25 October 2026

The current design update adds integrated memory/learning/vision/RSI ownership,
official Spark FP8 plus independent BF16 qualification, and a 34-source dataset
curriculum with 22 immutable metadata/card pins and two proposed internal corpora.
It revises the loop prompt, runbook, entry points and descriptive status registry.
Remaining R1–R3 code fixes are planned, not implemented; no weights, dataset
assets, native runs or new training are produced by this update. The 50 GiB
native policy and historical source/protocol/result files remain unchanged.

Validation covered local Markdown targets, balanced code fences, the three
changed/new JSON records, source counts/pins and preservation of rejection and
resource-policy fields. `git diff --check` passed. No runtime suite was rerun for
these documentation/descriptive metadata edits; review test counts above retain
their original scope. The workspace already contained unrelated changes, which
were preserved; no Git commit or staging was performed.

## Previous review and documentation handoff — 24 October 2026

Reviewed HEAD `988a46b` and its nine commits after `280d4dd`, current dirty source
and documents, affected implementation and native evidence. Independent token
replay reproduces all nine failed gates in
[the canonical verdict](reports/memory_reliability_v2_attempt03/verdict_final_v2.json).
The read-only review reran the 37 focused reliability tests (7.36 s); it inspected
but did not rerun the recorded 1,299-test / 21-subtest full suite.

Reproduced open defects: clean tracked HEAD source cannot import
`replica_lab.reliability_scoring` because `answer_audit.py` is untracked;
`verify_context` accepts altered span text with the original selection/digest.
The v1 builder and test file are also untracked protocol dependencies. Existing
native evidence is not shown to be tampered with; the consumer verifier needs
stronger checks before reuse. The source package, span verification, routing
accounting and prospective reference gate remain unmodified executable code.

The training schedule contains 64 gate-off and 128 gate-on cases; its native
counter says 0 gate-off. General retention has four distinct arithmetic queries,
and each evaluation split has four independent families. The earlier E4
contextual writer also failed 0/8 heldout; next diagnose binding/readout/copying
rather than repeating contextual features or adding steps without a hypothesis.

The user requested documentation and loop-prompt updates based on this review.
Documents 15–18 and the entry points now order repairs before MiniCPM5 visual
alignment/instruction tuning, visual memory and bounded RSI, with Spark later.
MiniCPM-V is an optional pretrained reference/teacher. This revision changes
Markdown only: it implements none of the reviewed code fixes, starts no training
or acquisition, and preserves all old protocol pins, reports and verdicts.

Documentation validation: 16 updated/new Markdown files checked, all 335 local
link targets resolve, fenced blocks balance and `git diff --check` passes.
No runtime test suite was rerun for this documentation-only change; the review
and saved full-suite results above retain their separate scopes.

**Next action:** document 17's reproducibility/context/scorer package, then one
diagnosed memory repair. Once that package has a measured outcome, isolated
V0–V2 vision may proceed with memory disabled even if learned-memory quality
remains rejected. Learned V3 memory and integrated V4/M5 RSI require their own
prerequisite quality and retention gates. Do not promote a failed dependency.

## Historical recovery records

The dated outcomes below are preserved evidence, not the current task queue.

## Reliability recovery outcome — 24 October 2026

**Completed and rejected:** [full report](docs/experiments/20261024_Memory_Reliability_Recovery_Results.md),
[final independent verdict](reports/memory_reliability_attempt02/verdict_final.json).
MiniCPM5 D1 trained 7,356,418 fast writer/reader parameters for 24 updates on
96 examples; all original parameters and non-fast extension tensors are unchanged.
Actual Muon int8-momentum scope was 7,340,032 parameters, plus 16,386 auxiliary
AdamW parameters. One isolated extension was saved; selected defaults are unchanged.

Benchmark/scorer repair now rejects cached-score tampering, wrong gold, missing
or duplicate cases/arms and inconsistent token decoding. Fresh counterfactual
multi-query data reduces the last-value shortcut to 9/48 fact queries. The
historical M4 audit verifies 434 nested answer rows including duplicate restart
references/external controls; its rejection and generation accounting remain intact.

The candidate scores **0/16 initial facts, 0/4 corrections, 0/12 unaffected facts,
8/8 absent facts and 0/8 general answers**. Disabled general answers score 8/8;
full-context control scores 48/48 overall. Both evaluation splits fail quality.
The independent verdict rejects 12/17 checks; no neural/RSI promotion is admitted.

All eight depth images authenticate exact scope, producer, source lineage and
2→3 counters. Fresh-process reload reproduces **7/7 token sequences and state
digests**, without training or source replay. This proves the tested persistence
mechanics, including persistence of incorrect answers; it does not prove useful
memory, crash/power-loss recovery, revocation or unlearning.

The actual native pilot took **115.35 s**, sampled driver peak **11.0043 GiB**;
restart took **6.01 s**, peak **10.0163 GiB**. Both observed zero swap growth and
passed the unchanged 50 GiB guard. Initial restricted-context attempt01 stopped
at host telemetry before model loading; attempt02 is the sole training run.
No base download/copy, protected-pilot modification or Git commit occurred.

Validation: **19 new recovery tests passed**; full suite **1,280 passed,
21 subtests passed, one restricted-context MPS skip**, 151.31 s, one existing
sparse-tensor warning. [JUnit](reports/validation_memory_reliability_20261024.xml).
The protocol is timestamped/hash-pinned, not committed, at Git base `280d4dd`.

Next: durable exact-record retrieval/correction reference and crash/revocation
coverage; a discriminating query-selectivity/read-interference diagnosis; then
a newly declared bounded D1 experiment on independent families. Reuse v1 only
as exposed regression evidence. Do not restart the same fine-tuning recipe or
advance RSI from mechanical success.

## Exact-reference and reader-only outcome — 24 October 2026

**Final decision: reject the learned candidate and reject the reference bundle on
full-context parity.** See [the experiment record](docs/experiments/20261024_Memory_Reliability_Reader_Only_Results.md)
and [canonical verdict](reports/memory_reliability_v2_attempt03/verdict_final_v2.json).
Selective exact context is **192/192**; full context is **190/192** because two
absent-fact answers are `orange`/`green` rather than `unknown`. The parity gate
therefore rejects the bundle without negating selective exact-reference utility.

The reader-only D1 arm completed **48 updates over 192 fresh training cases** and
changed exactly eight reader tensors (2,105,344 parameters). It scores **64/192**
fast answers: 32 unknown plus 32 general, with **0/64 fact, 0/16 changed and
0/48 unaffected**. Both splits independently fail all fact/correction floors and
delayed-above-controls checks. Parent-disabled behavior is unchanged, so factual
noninterference checks are vacuous. General/unknown retention passes.

All 32 images and source bindings authenticate, including training-inclusive
epoch 224. Fresh-process restart reproduces 7/7 answers and source bindings with
zero updates/checkpoints. Guarded wall time is 515.48 s; restart is 8.53 s;
sampled driver peak is 11,815,763,968 bytes with zero swap growth and 76% minimum
host-free memory. Protocol v4 is committed at `9a43bad`; scorer/finalizer fixes
are at `6b29460`. The final verdict separates native and smoke-fixture hashes and
records the gate-off accounting erratum: independently recomputed 64 versus
native counter 0. The counter is unauthenticated, not evidence of a gate-on-only
objective.

**Next:** inspect per-fact writer/state bindings before considering a new writer
arm. Reader-only failure does not establish correct storage. Do not add updates,
increase KL, switch models or promote RSI. Treat full-context absent-fact
hallucination as a separate abstention diagnostic.

Post-package full integration passed **1299 tests and 21 subtests in 137.53 s**
(zero failures/errors/skips), with 37 focused reliability tests also passing.
See `reports/validation_memory_reliability_reader_20261024.xml`.

## Preserved snapshots before the reliability recovery

These dated records preserve prior results; document 17 controls the next work.


**M2/M3 verified packages (24 October 2026 UTC).** M2's shared-value SGDN-2
diagnostic passed 4/4 dev and 4/4 held-out after context removal, above linear
(1/4 and 2/4) and all null controls, with correction and restart. Its candidate
retention is **0/2** (`amber`) versus parent 2/2, an ungated recipe failure
inherited by M3. The unique-value arm memorized training values and was rejected.
M3 then passed
`accept_m3_ownership_interfaces`: shared and private both scored 8/8 dev+held-out,
with independent per-depth writers, a defined convex shared reducer, update-two
historical gradients, gradient-cut, authenticated shared/private images and two
fresh-process restarts. Shared/private answers agreed 8/8. Private uses 131072
state bytes versus shared 65536, so no capacity-matched superiority is claimed.
Peak driver was 11.0043 GiB and selected training took 215.896 s. See
[M2 results](docs/experiments/20261024_M2_Dynamic_Memory_Results.md),
[M3 results](docs/experiments/20261024_M3_Depth_Ownership_Results.md) and
[the verdict](reports/m3_depth_comparison_attempt02/comparison.json).

**M3 repair and M4 static arm rejected (24 October 2026 UTC).** The M3 shared
replay candidate preserves fresh delayed 6/8 and original regression 8/8,
improves final general correct from 0/4 to 1/4 (`Cl`), but leaves development
0/2 and adversarial memory 0/4 versus disabled 2/4. Novel-value delayed cases
remain 0/2; seen-value is 4/4 and changed-template 2/2. It is not a parent.
M4 v7 used four source-LM plus 16 delayed updates and two extension saves.
Direct pattern is 0/4 dev, 0/8 held-out and 0/8 regression; every delayed router
row selected fast with incomplete hash coverage. Fast+pattern adversarial is
0/4 versus disabled 2/4. The independent v7 verdict rejects four quality gates;
16/20 total checks pass, including all listed engineering, correction and
same-process extension/image restart checks. Fresh-process restart was not run
because the candidate rejected. A CPU audit validated eight numeric collision
fields per group across 51 hash-call prompts and found zero distinct-n-gram
collisions in training and all-router scopes; repeated lookups are separate. See
[M4 results](docs/experiments/20261024_M4_Pattern_Results.md),
[final v7 verdict](reports/m4_static_attempt07/verdict_final.json) and
[collision audit](reports/m4_static_attempt07/collision_audit.json). Retention
and actually routed covered-case utility remain unproven.
Actual optimizer groups contain 34,615,300 pattern parameters; the finalized
verdict authenticates the original rejection by SHA-256. Latency is only a
single-prompt pattern-port microbenchmark, and covered-case utility is
unmeasured because no delayed route was fully covered.

Post-M4 verification passed **1262 tests plus 21 subtests in 133.73 s**
(1283 JUnit entries, zero failures/errors/skips), with 9 focused M4 tests also
passing. See `reports/integration_m4_v7_20261024.xml`.

**M0a/M1 verified packages (24 October 2026 UTC).** Corrected inference is
detached: the small point passed 2/2 and selected 4096 passed 12/12 with 28
nonzero backward gradients per point, zero swap growth and **28.5944 GiB**
sampled peak; 8192 was not rerun. Old-case menu regression is 3/3, but fresh
canonical is 1/4 and shuffled 0/4 due distractor selection; random 4/4 is not a
valid baseline (8 premature false claims). The deterministic procedure passes
3/3. M1's thin procedure compiler now executes, publishes, reconstructs after
restart, rejects changed sources/parents, revokes descendants and rolls back;
14 focused tests pass. See
[M0a results](docs/experiments/20261024_M0a_Refreshed_Baselines.md) and
[M1 results](docs/experiments/20261024_M1_Procedure_Compilation_Results.md).

**Current plan and guard update (23 October 2026).** Reviewed code HEAD
`377718e`, branch `codex/d2-capacity-runtime-pilot`. This revision updates
documents/status and the sampled driver guard default from 40 to **50 GiB**.
It runs no native model experiment and makes no new quality or capacity claim.
The last recorded full suite is **1211 passed / 21 subtests in 147.24 s**;
the current focused validation is **81 passed in 23.87 s**.

Changed executable scope: `CapacityPolicy.driver_limit_bytes` is now
53,687,091,200 bytes; both preflight CLI defaults derive from that policy.
Tests cover below/at-limit behavior, late-exit telemetry and explicit 40 GiB
overrides, plus guard-dependent validation/scoring contracts. Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_capacity_guard.py tests/test_generation_guard_runner.py tests/test_d2_validation.py tests/test_score_infrastructure_generations.py
```

Swap-growth 512 MiB, minimum host-free 15%, three consecutive host samples,
initial-pressure refusal, one native worker and no CPU fallback remain. No
allocator/OS settings changed. Historical explicit 40 GiB protocols/reports and
the pinned D2 resume launcher keep their policy. New 50 GiB experiments require
new protocol/code pins and fresh evidence; old guard failures remain failures.

The authoritative construction order and acceptance gates are in the
[research cross-check and backbone plan](docs/architecture/15_Research_Crosscheck_and_Backbone_Plan.md).
Use the [loop-agent instructions](docs/architecture/16_Loop_Agent_Instructions.md)
and [resumption record](docs/experiments/20261023_Resumption_Record.md) to resume.

**Current implementation and evidence.** Native memory/workspace/task paths,
exact journals, bounded tools and independent verification are connected.
Mixed completion is 4/4 with and without the action-kind head (no head benefit);
restart passes 12/12 acceptance checks. Free-form action learning remains
negative. The success head is diagnostically fitted but uncalibrated and has
no success authority. Native identity conditioning is attached but untrained.
E4 already has differentiable single-page learned-writer experiments, both
0/8 held-out; useful multi-chunk learned memory is open. Procedure compilation,
execution, single-writer atomic publication, rejection and rollback exist.
General compiler transactions and integrated neural self-improvement are open.

**Corrections already committed in `6e8ec34`.** Inference writer/sense now
run under `torch.no_grad()`; numeric-string constants enter action menus;
the scheduler rejects requests invalidated before admission. The historical
4096-unit 30.59 GiB capacity and 8192-unit 48.08 GiB guard boundary predate the
inference fix. Historical repair ranking of 17/17 (15 held-out) used the old
candidate menus. Corrected capacity and fresh matched menu controls remain
unrun. Retrieval/answer quality was measured separately at 16 units.

**Model decisions:** keep MiniCPM5 as the baseline. Optional M0b compares
Spark-X2.6-4B BF16 with a separate native adapter, measured Muon/FP32 residency,
reasoning/action quality and retention. MiniCPM-V-4.7-1B supplies the pretrained
vision path; 35B-A3B is a later qualified teacher/inference or transfer option.
M4 adds only a diagnostic static pattern branch and deterministic routing; no
production memory, learned router or RSI capability is established.

**Dependency disposition and next work:** M0b is dropped as an optional,
unadmitted track: no local Spark BF16 source, compatible isolated environment or
adapter exists, and it does not block MiniCPM5. M5 is blocked by the M4/M3
quality and retention failures and by C13's 5 accepted / 8 rejected-or-deferred
groups versus 10/10; a new admitted parent/data/control protocol and real
candidate comparisons are required. M6 remains blocked until a simpler loop
survives retention and noninterference. M7 remains blocked on local
MiniCPM-V-4.7-1B weights/revision, a separate adapter and native multimodal/cache
parity; larger GGUF artifacts do not satisfy that target. Personality contracts,
branch isolation, prompt/native-reader wiring and fictional evaluation are
implemented, but authentic training/fidelity awaits permitted decision records,
reserved families and human review. Do not relabel these blockers as completed
packages or promote M4 into M5.

## Historical execution entries

Entries below preserve the state when their work packages were recorded.
Their test counts, “current/next” language and resource measurements are
historical; use the snapshot above and the final next-gate section to resume.

**Current work — integrated E2/E3/C13/C16/C17 with measured boundaries (23 October
2026 UTC).** The shared width-256/eight-slot/two-iteration core, independent
memory/workspace readers, exact task journal, typed actions, real fixture tools,
independent success check and fresh-process restart are connected. On the mixed
family, **4/4 head-conditioned and 4/4 no-head episodes completed** with correct
repairs/passing evaluators, 0 malformed actions and 0 false success claims. The
action-kind head is rejected (no measured benefit). Fresh-process continuation
passes **12/12**: process 1 observed exit 1 and stopped; process 2 resumed the
same journal/ledger, repaired once, passed the evaluator and did not replay
completed actions. The success head has a **diagnostic fit** on 18 actual
outcomes but remains **uncalibrated, unpublished and without success
authority**; no general capability, confidence or personal-fidelity claim. The
sealed source is unchanged and no native job is active.

**E2 reader candidate (23 October).** A D1 candidate trained `record_writer` and
both memory readers with answer loss on 24 source-separated cases. Loss fell
but dev+held-out exact accuracy remained **0/16** for parent and **0/16** for
candidate; the candidate emitted training values rather than copying held-out
values despite correct lexical retrieval. Decision: **defer**, no default
change. This rules out “no answer-loss training” as the sole cause and points to
reader/writer representation or supervision; the 10 held-out cases must not be
retuned. See
[reader candidate results](docs/experiments/20261023_E2_Reader_Candidate_Results.md).

**Memory answer facade (23 October).** `MemoryAnswerService` connects the
opt-in pointer and exact-source evidence without generating a replacement
answer: pointer results carry record/span/source provenance; disabled or
unavailable pointer paths return exact source spans, never a host-chosen value.
Behavioral pointer metrics are unchanged. See
[memory answer path](docs/architecture/12_Memory_Answer_Path.md).

**E2 cross-template reranker (23 October).** Two fresh reranker candidates
were rejected: 4/8 held-out top1 after 8 updates and **1/8** after 32 updates,
against lexical **5/8**; the higher-budget arm overfit while training loss fell.
Lexical remains default; the existing reranker remains opt-in. See
[cross-template rerank](docs/experiments/20261023_E2_Cross_Template_Rerank_Results.md).

**E2 8192-unit guard boundary (23 October).** A single-point 8192-unit probe
with the pre-fix inference harness hit the capacity guard during source-writer
processing at **48.08 GiB** sampled peak, above the 40 GiB policy; no update or
checkpoint. The selected-weight maximum remains **4096 units**; 8192 is measured
as a boundary, not a pass. The harness now runs writer/sense under
`torch.no_grad()`; corrected inference must be remeasured before treating 48.08
GiB as the configuration's limit. See
[capacity results](docs/experiments/20261023_E2_Retrieval_and_Capacity_Results.md).

**E2 4096-unit selected capacity (23 October).** The selected extension passed
all 12 inference/modules-only backward points on a fresh 4096-unit bank: peak
driver **30.59 GiB**, **559.4 s**, zero swap growth, 28 gradient tensors. No base
save/optimizer update. Tested ladder: 32/64/128/256/512/1024/4096; 8192 was
attempted separately and stopped by the guard, not qualified. The 4096 peak
predates the no-grad correction; the pass remains valid for its workload, while
the true inference peak needs remeasurement. Capacity only. See
[capacity results](docs/experiments/20261023_E2_Retrieval_and_Capacity_Results.md).

**E2 1024-unit selected capacity (23 October).** The selected extension passed
all 12 inference/modules-only backward points on a fresh 1024-unit bank: peak
driver **16.90 GiB**, **142.6 s**, zero swap growth, 28 gradient tensors. No
base save/optimizer update. Tested ladder: 32/64/128/256/512/1024 units;
capacity only. See
[capacity results](docs/experiments/20261023_E2_Retrieval_and_Capacity_Results.md).

**E2 512-unit selected capacity (23 October).** The selected extension passed
all 12 inference/modules-only backward points on a fresh 512-unit bank: peak
driver **14.62 GiB**, **76.0 s**, zero swap growth, 28 gradient tensors. No base
save/optimizer update. The tested capacity ladder is now 32/64/128/256/512
units; capacity only, no quality claim. See
[capacity results](docs/experiments/20261023_E2_Retrieval_and_Capacity_Results.md).

**E2 256-unit selected capacity (23 October).** The selected extension passed
all 12 inference/modules-only backward points on a fresh 256-unit bank: peak
driver **13.50 GiB**, **43.4 s**, zero swap growth, 28 gradient tensors. No base
save/optimizer update; capacity only. See
[capacity results](docs/experiments/20261023_E2_Retrieval_and_Capacity_Results.md).

**E2 128-unit selected capacity (23 October).** The selected extension passed
all 12 inference/modules-only backward points on a fresh 128-unit bank: peak
driver **12.93 GiB**, **27.2 s**, zero swap growth, 28 gradient tensors. No base
save/optimizer update; capacity only. See
[capacity results](docs/experiments/20261023_E2_Retrieval_and_Capacity_Results.md).

**E2 64-unit selected capacity (23 October).** The selected extension passed
all 12 inference/modules-only backward points on a fresh 64-unit bank through
the validated `max_units`-only migration: peak driver **12.69 GiB**, **19.1 s**,
zero swap growth, 69% minimum free, 28 gradient tensors. Capacity only; no
quality or longer-source-length claim. See
[capacity results](docs/experiments/20261023_E2_Retrieval_and_Capacity_Results.md).

**E2 32-unit capacity (23 October).** A fresh 32-unit bank passed all 12
inference/modules-only backward points under the guard. The architecture-only
arm (random ports, no parent) used 12.57 GiB and 17.4 s. A validated
**port-capacity-only** migration then loaded the selected E2 extension with
identical tensors and only `ports.max_units` increased to 32; it also passed
12/12, at **12.57 GiB**, **14.9 s**, zero swap and 28 gradient tensors. No base
save or optimizer update. This qualifies selected-weight capacity at 32 units
for the synthetic bank; it is not a quality or longer-source claim. See
[capacity results](docs/experiments/20261023_E2_Retrieval_and_Capacity_Results.md).

**E2 cross-template retrieval (23 October).** Lexical top-1 **7/12**, native
dense **5/12**, and the existing opt-in reranker **1/12** on the cross-template
family; all had recall@4 12/12. The reranker does not transfer to the new
phrasing and remains opt-in; lexical remains default. See
[cross-template retrieval](docs/experiments/20261023_E2_Cross_Template_Retrieval_Results.md).

**E2 cross-template boundary (23 October).** A fresh varied-phrasing family
had lexical top-1 **7/12** (recall@4 12/12), so the top-record context test is
confounded. Parent pointer scored 3/12 and context candidate 6/12 among the
accessible top records; no broad template or default claim is made. See
[cross-template results](docs/experiments/20261023_E2_Pointer_Cross_Template_Results.md).

**E2 context head holdout 2 (23 October).** The published opt-in context head
repeated **12/12** versus parent **9/12** on a second untouched 12-case family;
parent misses were entity-name spans. Provenance/manifest checks pass. The
feature set remains template-specific and exact text remains the default. See
[holdout2 results](docs/experiments/20261023_E2_Pointer_Context_Holdout2_Results.md).

**E2 integrated context head (23 October).** The saved 6-parameter context
head was integrated into `MemoryAnswerService` and evaluated on a new family:
**12/12** exact versus parent **9/12**. It fixed entity-name span errors. The
candidate was published in a separate opt-in manifest, a harmful candidate was
rejected, and rollback to exact text passed. Exact text remains the default; the
template-specific feature set is not a broad generalization claim. See
[context service results](docs/experiments/20261023_E2_Pointer_Context_Service_Results.md).

**E2 pointer context diagnostic (23 October).** A 6-parameter candidate-context
head over frozen pointer scores scored **12/12** on a new family versus **9/12**
for the parent pointer, fixing entity-name span errors. It is a diagnostic
artifact only: not loaded by production `MemoryAnswerService`, not published as
an active manifest, and not a default. See
[pointer context results](docs/experiments/20261023_E2_Pointer_Context_Results.md).

**E2 pointer top-record continuation (23 October).** Continuing only the copy
head on top-record whole-word candidates produced **8/12** on a new family,
identical to the parent (**8/12**), with flat training loss and identical spans.
Rejected; pointer remains opt-in and exact text remains default. The remaining
error is within-record entity-name span selection. See
[scope continuation results](docs/experiments/20261023_E2_Pointer_Scope_Train_Results.md).

**E2 pointer record scope (23 October).** On a third untouched family, the
same pointer weights with whole-word candidates scored **10/12** when restricted
to the top lexical record versus **3/12** across all selected records. The two
misses were entity-name spans inside the correct record. This identifies source
selection as the dominant fresh-copy bottleneck; it remains opt-in and exact
text remains default. See
[record-scope results](docs/experiments/20261023_E2_Pointer_Record_Scope_Results.md).

**E2 pointer candidate units (23 October).** On a second untouched family,
token and whole-word candidate enumeration both scored **2/12**. Whole-word
candidates removed truncated copies (`saff`→`saffron`) but still selected wrong
source values, so query/source scoring is the bottleneck. The first attempt
failed on punctuation-gap spans exceeding the bound; the builder now requires
contiguous tokens. No training; opt-in only. See
[pointer units](docs/experiments/20261023_E2_Pointer_Units_Results.md).

**E2 pointer fresh consumer (23 October).** The published pointer was evaluated
on 12 untouched-value cases without training: **3/12** exact, complete
provenance, exact-source fallback pass. Misses include partial subword copies
(`saff`, `ilion`, `cer`) and distractor selections. It remains opt-in; exact text
is the default. See
[fresh pointer results](docs/experiments/20261023_E2_Pointer_Fresh_Service_Results.md).

**E2 pointer consumer (23 October).** `MemoryAnswerService` was exercised on
the published pointer extension: pointer dev **3/6**, held-out **2/10**, and all
16 requests had complete record/span/source-hash provenance; the exact-source
fallback also passed. Two API defects were fixed: unwrapping `ExactSpan.text`
and separating record IDs from source hashes. No training, checkpoint or default
change. See
[pointer consumer results](docs/experiments/20261023_E2_Pointer_Service_Results.md).

**E4 permitted replay (23 October).** After correction/revocation, the host can
reset fast memory and replay a bounded sequence of still-permitted
`(observation_id, text, source_hash)` observations. Runtime replay clears
dependent workspace, updates live fast state, and records a host correction note;
no generated answers enter the replay. This is reconstruction plumbing, not
learned delayed-use benefit. See
[fast-memory reference](docs/architecture/10_Fast_Memory_Reference.md).

**E4 source invalidation (23 October).** `TaskRuntime.invalidate_source` now
records a host-authority correction note, clears dependent workspace, and resets
live fast state when the source hash appears in the committed fast-memory
lineage; unmatching sources leave fast state intact. Persisted images remain
historical evidence and must be rebuilt after correction, not treated as current.
See [fast-memory reference](docs/architecture/10_Fast_Memory_Reference.md).

**E4 image reload (23 October).** A committed fast-memory image now loads into a
fresh `FastMemoryBridge` with exact owner/scope/producer validation, reconstructs
its observation count, rejects cross-branch/tampered images, and can seed the
next runtime decision from the restored snapshot. The bridge's duplicate
directory creation was fixed. This is persistence plumbing, not learned
delayed-use benefit. See
[fast-memory reference](docs/architecture/10_Fast_Memory_Reference.md).

**E2 pointer publication (23 October).** The opt-in pointer candidate was
registered as an accepted neural candidate, a deliberately harmful duplicate
was rejected, and a separate opt-in manifest was published and rolled back to
the exact-text baseline. This is metadata/publication only; the pointer is not
the default and has no runtime promotion. See
[pointer publication](docs/experiments/20261023_E2_Pointer_Publication_Results.md).

**E4 contextual writer (23 October).** A new source-feature writer reduced
training loss **50.1→8.3**, but train exact was **3/24**, held-out **0/8**, and
shuffled/zero/disabled all 0/8. State influence is present; reliable novel-value
copy is not. Rejected; exact text remains preferred. See
[contextual writer results](docs/experiments/20261023_E4_Contextual_Writer_Results.md).

**E2 exact-span pointer (23 October).** A pointer over exact selected-record
spans produced **3/6 dev** and **2/10 held-out** correct novel values, with
**0/6 and 0/10** for shuffled source spans. It emits source token IDs directly
without LM generation. The preregistered rule accepts it **opt-in only**; exact
text remains preferred. Training loss was flat (~18.5), so this is a modest
candidate, not general reader capability. See
[pointer results](docs/experiments/20261023_E2_Pointer_Copy_Results.md).

**E2 record-token copy head (23 October).** Added optional exact-token
`CopyRead` frames and an independently gated copy head, plus loader/serialization
paths. Trained only the copy head on fresh novel-value data; copy-enabled and
disabled answers were identical, held-out accuracy **0/16** both, and training
loss did not descend. The interface tests pass; the candidate is **rejected**.
See [copy-head results](docs/experiments/20261023_E2_Copy_Head_Results.md).

**E3 pooled success head (23 October).** The runtime summary contract is 24
task features plus one pooled 256-wide workspace; the old full-slot reshape was
replaced with a matching pooled monitor. Trained on 18 actual outcomes (14
positive / 4 negative), case-held-out accuracy is **0.8125** vs a **0.7778**
majority and Brier **0.342**, with a 0.25 failure fold on
`prompt_heldout_repair_01`. It is **uncalibrated and unpublished**, with no
success authority. See
[success-head results](docs/experiments/20261023_E3_Success_Head_Results.md).

**E3 core controls (23 October).** On one fresh repair/pass pair with exact
constant evidence, the full core/workspace path completed **2/2**, while
core-disabled, feed-forward and extra-inference controls each completed only the
pass case (**1/2**), failing the repair with rejected false stop claims. The
signal is bounded to n=2 and does not establish recurrent-core superiority.
Attempt 1 used a 640-token cap and failed after full episodes; preserved. See
[core-control results](docs/experiments/20261023_E3_Mixed_Core_Controls_Results.md).

**E3 exact-constant prompt evidence (23 October).** A host-read line now lists
the current workspace constant names/current values (never the expected bound or
a proposed repair) when `set_constant` is allowed. On a new four-case family
this produced **4/4 completions**, 0 malformed actions and 0 false success
claims; the model still selected the name/value. This is a narrower evidence
interface, not a general capability claim. See
[prompt evidence results](docs/experiments/20261023_E3_Prompt_Evidence_Results.md).

**E3 fresh mixed family (23 October).** Four new held-out cases with constant
names/values absent from accepted training produced **3/4 completions** with no
head. The miss was `fresh_heldout_repair_02`: the model generated `12` but
misspelled the new constant name (`QUUEUE_DEPTH`/`QUUE_DEPTH`); the host rejected
both mutations and both stop claims. This is an argument-generation boundary,
not a tool/parser failure. See
[fresh mixed results](docs/experiments/20261023_E3_Fresh_Mixed_Controls_Results.md).

**E3 mixed fresh-process restart (23 October).** Process 1 read and observed
evaluator exit 1, then stopped at the declared step limit; process 2 resumed the
same journal and ledger, applied exactly one `set_constant(POLL_ATTEMPTS=7)`,
passed the independent evaluator, and did not replay completed actions. All
**12/12** acceptance checks pass on one copied synthetic case. Attempt 1 used
`mode=full` and repeated `read`; it is preserved. See
[restart results](docs/experiments/20261023_E3_Mixed_Restart_Results.md).

**E3 mixed-outcome policy head (23 October).** On the fresh four-case mixed
family, actual model actions completed **4/4** held-out episodes under
kind-conditioned generation and **4/4** no-head controls; repairs were applied
and the independent evaluator passed, with 0 malformed actions and 0 false
success claims. Head and control action sequences were identical, so the head
has **no measured benefit** and is rejected as a candidate. This is a bounded
synthetic success, not general capability; no default was changed. See
[E3 policy-head results](docs/experiments/20261023_E3_Policy_Head_Results.md).

**E3 mixed-outcome probe (23 October).** A mixed pass/repair family has
different labels at the post-test step. A 64-hidden probe on prompt-only core
states reaches **1.00** held-out action-kind accuracy versus **0.875** step-only
and **0.375** class-majority; this is representation information, not generated
actions or task completion. No policy head was deployed. See
[mixed probe results](docs/experiments/20261023_E3_Mixed_Policy_Probe_Results.md).

**E3 policy probe (23 October).** Prompt-only core states from eight actual
teacher trajectories classified next action kinds at **100%** for joint,
workspace-only and task-only features. All trajectories share the same
five-step action order, so step position predicts the label; the result is
**inconclusive/evaluation ambiguous**, not a representation or policy success.
No policy head was deployed and no default changed. Attempts 1/2 were preserved
as implementation defects. See
[E3 policy probe results](docs/experiments/20261023_E3_Policy_Probe_Results.md).

**E4 delayed-use candidate (23 October).** A learned token-feature fast writer
plus native readers at blocks 13/27 was trained for 32 modules-only updates on
24 source pages. Held-out exact accuracy is **0/8**; train is 7/24; correct and
shuffled states diverge on 6/14 cases, while zero state equals disabled 14/14.
A no-retraining compositional diagnostic with held-out entities and seen values
scores 1/8 correct and 1/8 shuffled, 0/8 zero/disabled. The writer/representation
does not generalize; the arm is rejected, no default change. Persisted-image
reload is exact but is same-process. See
[E4 delayed-use results](docs/experiments/20261023_E4_Delayed_Use_Results.md).

**E5 neural candidate (23 October).** The existing E2 reranker extension was
evaluated against its parent on source-separated retrieval cases with no new
training: held-out top-1 **0.25 → 0.50** (parent → candidate), but lexical is
**0.75**. The preregistered rule accepts the candidate relative to its parent;
deployment is **opt-in only**, lexical remains default. Dev top-1 is 0/2 for
both. This is a real parameter-level improvement on unseen sources but not a
useful default or task-capability result. See
[neural candidate results](docs/experiments/20261023_E5_Neural_Candidate_Results.md).

**E5 procedure candidate (23 October).** A host `RepairProcedure` compiled from
four verified train trajectories succeeded on **3/3 held-out + 1/1 dev** cases
with independent evaluator receipts, preserved non-target constants, and was
published through `ManifestRegistry`; a deliberately harmful candidate was
rejected and rollback to an ancestor was tested. This is a procedure-library
baseline, **not** a neural parameter update or model-selected action. It shows
the task family is solvable while the model action policy remains 0/7. See
[procedure results](docs/experiments/20261023_E5_Procedure_Results.md).

**E4 typed artifacts (23 October).** ContextPack, RecordLatent,
BranchMemoryImage, ProcedureArtifact and NeuralCandidate metadata contracts now
have kind-specific lifecycles, source/parent dependency tracking, expiry,
payload rejection and transitive invalidation. The registry prevents a view
artifact from being accepted like a trained candidate and keeps proposed
candidates from self-certifying. No artifact execution, candidate training,
independent acceptance or transactional publication exists; the next step is a
bounded E5 candidate. See
[typed artifacts](docs/architecture/11_Runtime_Artifacts.md).

**E4 reference fast memory (23 October).** A functional linear host service now
supports observation-only staged writes, one deferred commit against a single
chunk-start state, per-owner/branch scope, bounds, lineage, reset/invalidate and
strict safetensors image persistence. Focused software tests cover invisibility,
idempotency, shuffled/reset controls, bounds and cross-branch/producer/tamper
rejection. Opt-in native read ports are attached at 13/27 and pass null-parity/independent influence (0.269 logit change, 11.0 GiB peak). An optional bridge now stages/commits actual read/test observations and passes the committed state to the next decision; the encoder is deterministic and **untrained**. No learned write/query training, delayed-use task study, automatic image reload or interference evaluation exists. See
[fast-memory reference](docs/architecture/10_Fast_Memory_Reference.md).

**E3 candidate reranker integration (23 October).** Menu construction and model
scoring moved into the reusable `runtime/candidate_actions.py` component; the
experiment runner now calls it. A post-refactor smoke reproduced **6/6 (5/5
held-out)** with zero false success claims at 15.12 GiB, 66.5 s. This is a
refactor/verification, not a new capability arm. See
[candidate rerank results](docs/experiments/20261023_E3_Candidate_Rerank_Replication_Results.md).

**E2 independent answer scoring (23 October).** Removing the candidate-list
position artifact did not rescue answer selection: model **1/4** vs random
**0/4**, with distractors preferred in three cases. Candidate ranking remains
limited to the repair menu; general answer ranking is not qualified. See
[independent answer results](docs/experiments/20261023_E2_Candidate_Answer_Independent_Results.md).

**E2 answer candidate ranking (23 October).** The repair-family candidate
scorer was applied to fresh source-answer candidates. Canonical order scored
**4/4**, but the model always chose index 0; a shuffled-order model control
scored **0/4**, and random scored **0/4**. The canonical result was a position
artifact. Candidate ranking is not established outside the repair family. See
[answer candidate results](docs/experiments/20261023_E2_Candidate_Answer_Results.md).

**E3 candidate order control (23 October).** A fresh repair family scored
**3/3** with canonical and **3/3** with shuffled menus; selected repair indices
changed (4 vs 1), ruling out a fixed-position artifact. Peak 15.15 GiB, 31.6 s
per arm, no training. Across four fresh repair families: **17/17 (15/15
held-out)**; random 0/11. This does not generalize to source-answer ranking. See
[candidate order results](docs/experiments/20261023_E3_Candidate_Order_Results.md).

**E3 filtered candidate restart (23 October).** A fresh three-case family was
interrupted after two decisions per case (0/3), then resumed in a fresh process
with exact journal/workspace reuse; the filtered menu excluded values already
contradicted by the observed failure and completed **3/3** with no false success
claims. Process 1 **10.7 s**, process 2 **25.7 s**, peak **15.11 GiB**. This
strengthens restart integration for the opt-in ranking interface, not free-form
action selection. See
[filtered restart results](docs/experiments/20261023_E3_Candidate_Restart2_Results.md).

**E3 candidate-ranking restart (23 October).** Process 1 persisted two
decisions per fresh case after a failed test; process 2 reloaded the exact
journal/workspace and completed **1/2** cases. The successful case passed the
independent evaluator; the other selected the unchanged-value candidate,
produced one false stop claim and exhausted its continuation budget. This
verifies restart mechanics for the interface but not robust ranking. See
[candidate restart results](docs/experiments/20261023_E3_Candidate_Restart_Results.md).

**E3 candidate reranking replication (23 October).** A fresh 1-dev/5-heldout
family repeated the result: model **6/6 (5/5 held-out)** vs same-menu random
**0/6**, zero false success claims. Across two fresh families: **11/11 episodes,
9/9 held-out, vs 0/11 random**. Model peak 14.14 GiB, 66.6 s; no training. This
strengthens the opt-in ranking interface but remains narrow and does not rescue
free-form action selection. See
[replication results](docs/experiments/20261023_E3_Candidate_Rerank_Replication_Results.md).

**E3 candidate reranking (23 October).** A host-enumerated action menu with
the model scoring each candidate produced **5/5 completions (4/4 held-out)**
with zero false success claims; the same-menu random control produced **0/5**
with two false claims per episode. Model peak **16.23 GiB**, 56 s. This is an
opt-in narrower ranking interface, not a free-form action policy; the menu
contains the disclosed repair value and the family is small. Free-form action
results remain negative. See
[candidate-rerank results](docs/experiments/20261023_E3_Candidate_Rerank_Results.md).

**E3 DAgger aggregation (23 October).** Initial accepted E3 extension completed
**2/5** fresh held-out/dev episodes. After 8 modules-only aggregation updates on
26 host-teacher-labelled model states, it completed **0/5**; training loss fell
to **0.023** but post-training actions became malformed. Peak 16.21 GiB, 430 s,
extension 37.6 MB, no base save, no publication. This is a negative DAgger result;
the initial 2/5 is not a generalization claim. See
[DAgger results](docs/experiments/20261023_E3_Dagger_Results.md).

**E3 bf16 full-backbone D2 action (23 October).** A declared bf16-compute
residency variant completed 8 full-scope updates at **39.10 GiB** peak, then
completed **0/4 held-out and 0/2 dev** episodes; every episode repeated `read`.
No export was written. This closes the full-backbone action hypothesis as well
as the residency route for this interface. See
[D2 action results](docs/experiments/20261023_E3_D2_Action_Results.md).

**E3 D2 action residency (23 October).** The full-backbone action diagnostic
was stopped by the capacity guard at the first backward: **40.68 GiB** sampled
peak against a 40 GiB limit, with no optimizer update, checkpoint or held-out
evaluation. Teacher prompts reach 609 tokens, so a 512-token rerun would
truncate task context and was not run. See
[D2 action results](docs/experiments/20261023_E3_D2_Action_Results.md).

**E3 large supervised action family (23 October).** A fresh 12-train /
3-dev / 6-heldout family trained modules-only to teacher loss **0.0276**, yet
full-model episodes completed **0/9** and always emitted `read`; all controls
failed. Peak **16.78 GiB**, 470 s, 32 updates, 37.6 MB extension, no base save.
This closes the larger-supervision hypothesis for the current interface; action
representation/policy is the remaining failure. See
[large-family results](docs/experiments/20261023_E3_Large_Action_Family_Results.md).

**E3 constrained action decoding (23 October).** Adding token-level masking of
redundant `read`/`run_test` strings to the runtime guard was aborted at the
predeclared smoke: banning `read` produced `.read` and malformed/unauthorized
JSON, **0/1** completion, no repair selected, no false success. This is an
interface/decoding defect; the remaining cases were not run. See
[constrained-action results](docs/experiments/20261023_E3_Constrained_Action_Results.md).

**E3 action-space guard (23 October).** A general no-progress guard (deny a
repeated read of an unchanged file and a repeated test after failure) was tested
on a fresh four-case `set_constant` family. Guard and control both completed
**0/4**; the model emitted `read` repeatedly, with the guard denying reads but
not changing action selection. No false success, no repair selected. This closes
the prompt-only guard path and supports an action-selection, not interface,
diagnosis. See [guard results](docs/experiments/20261023_E3_Action_Guard_Results.md).

**Retention inventory (23 October).** A read-only manifest now validates the
declared retention policy, measures bytes/files and hashes small artifacts
without deleting or rewriting anything. It records the sealed source, selected
extensions, protected `runtime_pilot.py`, and rejected candidates; no deletion
was performed. Manifest refreshed for the 256–4096 capacity datasets, 8192 failure evidence and candidate controls: `reports/retention_manifest_20261023.json`. See
`configs/retention_policy_v1.json` and `src/replica_lab/retention.py`.

**C17 bounded cache manager (23 October).** An exact-ownership-key host cache
now binds producer/writer/tokenizer, record/span hashes, scope and identity, with
deterministic entry/byte bounds and source invalidation. It is not native
KV/recurrent reuse; the adapter still re-prefills. See
[request scheduler](docs/architecture/14_Request_Scheduler.md).

**C17 request admission (23 October).** A single-job `RequestScheduler` now
binds each admitted operation to request/task/revision/evidence/branch/producer
identity, refuses a second concurrent admission, and supports explicit host
invalidation. It is admission-only: the adapter still re-prefills, no native
cache is reused and existing native scripts remain serialized by the exclusive
worker lock. See [request scheduler](docs/architecture/14_Request_Scheduler.md).

**C13 checked-proposal gate (23 October).** A versioned registry now records
13 actual candidate comparisons with no synthetic labels. It has 13 independent
source groups, **5 accepted and 8 rejected/deferred**, below the declared 10/10
minimum, so the learned selector is deferred and no weights were trained. See
[learning director](docs/architecture/13_Learning_Director.md); report:
`reports/c13_checked_proposals_audit_20261023.json`.

**C13 actual-outcome audit (23 October).** Six historical summaries were
reviewed by the deterministic director: accepted procedure -> no update; large
action failure -> neural proposal; retrieval failure -> record proposal; smaller
action/DAgger/restart cases deferred for insufficient labels. This is a policy
audit, not learned-selector accuracy. See
[learning director](docs/architecture/13_Learning_Director.md). Report:
`reports/c13_director_audit_20261023.json`.

**C13 outcome-to-proposal adapter (23 October).** `runtime/learning_cycle.py`
now turns an actual runtime summary into one bounded director review, with
success→no-update, repeated-read/malformed→action and receipted failure→procedure
mapping. It has no publish, execute or goal authority. Four focused tests pass;
this is integration, not a trained director. See
[learning director](docs/architecture/13_Learning_Director.md).

**C13 learning director baseline (23 October).** A deterministic director now
aggregates actual train/development outcomes, diagnoses repeated failure classes,
and emits at most one bounded `LearningProposal`, a deferral or no update. It
requires verified receipts for procedure proposals, at least eight attempts for
neural proposals, excludes audit/gold, enforces cost/proposal caps and has no
publish/execute authority. It is **not a learned selector** and has no measured
learning benefit. See [learning director](docs/architecture/13_Learning_Director.md).

**C16 fidelity harness (23 October).** A fictional scoring harness now compares
generic, approved-summary and archive-retrieval baselines while keeping
provenance, choice, clarification, unknown, technical, style and branch
divergence separate. It rejects non-fictional cases and makes no fidelity claim.
Authentic data and human review remain external. See
[fidelity harness](docs/architecture/17_Replica_Fidelity_Evaluation.md).

**C16 persistent identity store (23 October).** The source archive, approved
snapshots, revocation history and branch-owned experience now persist in an
atomic, digest-bound, mode-0600 file; reload rejects tampering/unknown formats
and revoked snapshots cannot resolve. Identity remains optional/null and has no
verifier authority. Authentic source, native reader attachment and personal
fidelity remain pending. See
[identity interface](docs/architecture/09_Replica_Identity_Interface.md).

**C16 identity interface and native reader (23 October).** Versioned source
episodes, approved conditional principle hypotheses, dated snapshots, mode-bound
identity contexts, parallel-branch state, correction/revocation handling, a
digest-bound persistent store and a null path are implemented. The optional
zero-initialized `IdentityReader` is now attached after the shared core workspace
with an explicit enable gate and an `identity_digest` bound into `CoreState` and
workspace persistence; identity-bearing exports use manifest v3. Tiny offline
tests pass null/disable parity, all-masked slots, nonzero influence, session
plumbing, persistence and v3 reload. The selected core-only E3 extension remains
v2. No authentic identity slots, trained conditioning, runtime slot encoder or
personal fidelity evaluation exists. See
[identity interface](docs/architecture/09_Replica_Identity_Interface.md).

**E3 narrower repair arm (23 October).** A fresh `set_constant` task family
(four train/one dev/three held-out, disjoint groups) passed all engineering
checks: initial parity, E2 migration, both gradient paths, stale-state rejection,
extension-only export, and zero base copies. Teacher-forced loss fell to **0.015**
(recurrent) but held-out execution was **0/7**: full and feed-forward arms repeated
`read` seven times; core-disabled/extra-inference produced malformed proposals.
The model never selected `set_constant`, so this is action selection, not tool or
argument parsing. Peak driver **15.72 GiB**, zero swap; 32 updates; the success
head remained untrained (seven negative labels). Preserve this as the second
negative action-policy family; do not run another near-identical arm. See
[set_constant results](docs/experiments/20261023_E3_Set_Constant_Results.md).

**E2 follow-up — retrieval, novel records and capacity (23 October 2026 UTC).**
A source-separated 13-case/49-record retrieval set and a 16-unit capacity bank
were committed with protocols before execution. Held-out record recall@1 is
**0.75** for lexical and native dense, **0.50** for learned reranking; all slot
arms answer **0/6** held-out cases and the oracle correct-record diagnostic
answers **0/4**. Exact source text answers **3/3**; correct and swapped slots
produce the same old answer. The bottleneck is therefore record writer/reader
encoding, not eligibility, retrieval availability or the decoder. Eligibility
violations are zero, corrections/superseded records are filtered, and the long
record's fact span is included. Reranking remains **opt-in**; lexical remains
the default; dense is not preferred.

The largest actually qualified retrieval/answer path is **16 units / 2,043
source tokens / 256 prompt tokens / 8 generated tokens**, plus a modules-only
16-unit backward. This is the retrieval workload, not the later capacity
maximum; the selected extension was separately qualified through **4096 units**
(see the current snapshot above). All 12 schedule points completed; peak sampled
driver memory **12.51 GiB**, zero observed swap growth, 69% minimum free. D2
capacity was not declared or run.
See [E2 results](docs/experiments/20261023_E2_Retrieval_and_Capacity_Results.md).
Validation for this increment: full suite **1091 passed / 21 subtests**, E2
dataset tests 3 passed; receipt
[validation_e2_followup_20261023.json](reports/validation_e2_followup_20261023.json).

**Open:** held-out configuration repair, correct action selection/arguments,
full-core plan continuation, success-head training/calibration, novel-value
E2 writer/reader transfer, ranking quality, and a post-fix inference
remeasurement of the 4096/8192 capacity boundary. These are
exposed synthetic fixtures, not a generalization result. Do not retune this
held-out set. A future narrower repair-tool arm or new held-out cases must be a
new preregistered study.

Validation: the pre-E3 full suite passed **1081 tests / 21 subtests** after the
runtime/neural integration; the E3-focused suites pass **157 tests**. Native
attempts and all failures are preserved. Attempt 1 failed on post-migration
null-parity ordering; attempt 2 completed training but rejected every valid JSON
followed by `</s>` as `multiple_json_objects` (56/56 denied, 0 tools); attempt 3
fixed that parser and produced the accepted evidence. The interrupted pre-check
also preserves a nested-binding CLI failure. See
[E3 results](docs/experiments/20261023_E3_Native_Results.md).

**Progress commits:** E2 follow-up `dabaa00` protocols/data and `3fd5c03`
answer-path diagnostic; E3 `e16a3f3` core/readers, `59ef1bb` journal/actions/runtime,
`8e95e2b` preregistered protocol before execution, then focused parser/binding
fixes (`46f1b59`, `bfaa0e7`, `867fe30`, `e9eb7d6`). Native jobs are idle. Total
alignment work: 32 updates in attempt 2 + 32 in attempt 3; attempt 1 performed
no optimizer update. The protected untracked `runtime_pilot.py` remains
unchanged and excluded.

## Preserved E1/E2 native memory

E1/E2 remain implemented and GPU-qualified: slot-only **4/4**, ports disabled
**0/4**, exact fallback 4/4, source-swap controls and fresh-process
correction/reload pass, **21.11 GiB** peak driver memory and one **18.62 MiB**
extension at `runs/e1_e2_native_attempt02_20261023/selected_extension`. Reranker
top-1 remains 1/2. See [E1/E2 results](docs/experiments/20261023_E1_E2_Native_Results.md).

## Preserved latest native result — query-policy v2

22 October 2026, America/Chicago. Branch `codex/d2-capacity-runtime-pilot`.
The source-bound infrastructure release remains **2,000 independently verified
targets / 1,000 synthetic worlds**, four families, 1,200 train and 200 each dev,
calibration, promotion, audit. All roles are 50% answer, 25% ask and 25% unknown and fit 512
tokens. All role fixtures are exposed; this is a development infrastructure
release, not a sealed generalization benchmark.

**The simpler query-policy v2 run is complete as a measured negative.**
Protocol/code/data/rubric/judge plan were committed at `b2010c2` before the one
native run. Each query receives exactly its public missing-response policy,
selected independently of observed missingness and gold labels. Original model,
evidence, ordered 16 cases, parser, greedy seed 17, cap 32, max 512, thinking disabled and
MPS FP32 fallback disabled remain fixed. No adaptive native follow-up occurred.

Program task correctness and the independent fresh-context semantic subagent
both score **4/16**; independently derived expected answers match frozen gold
on 16/16 cases. All four successes are UNKNOWN. All eight complete cases also
return UNKNOWN and all four asks name observed fields. This equals the
constant-UNKNOWN control: the 4/16 score is not useful evidence-responsive gain.
Program grammar accepts 12/16; all 16 outputs use one line and EOS, with zero
caps. Judge instruction compliance is 4/16 because it also checks the correct
case-dependent branch; it is distinct from program grammar validity.

The original baseline remains 0/16 with 16 caps; v1 remains 0/16 with 1 valid grammar,
16 EOS and no caps. V2 changes surface response selection but leaves evidence
use and missing-field identification unresolved. Query-policy-conditioned
fallback/abstention is observed; a particular internal mechanism is unproven.
No arithmetic answer was attempted, so inability to calculate is not isolated.
No optimizer updates occurred, so this cannot diagnose Muon 8-bit. Sampling or
thinking changes require separately declared controls. The judge saw only its
frozen rubric and shuffled allowlisted packet, without gold/fixed scores/IDs;
one agent over reused exposed cases is not a calibrated independent audit.

Prompt tokens are 5,294 (original 3,946, v1 5,574); maximum prompt+reference 407,
without truncation. Outputs total 86 tokens and **4.412 s generation**; profiled
lifetime **9.031 s**, guard lifetime 11.418 s. Sampled peak MPS live/driver is
**9.427 / 10.096 GiB**. Process RSS sampled/lifetime is 3.967 / 14.491 GiB. Exact guard
replay passes: zero observed swap growth, minimum host-free 71%. Counters overlap;
0.25 s device / 2 s host samples can miss transients. Native inference used MPS;
ANE execution is not qualified by this backend.

The initial fixed-scoring attempt stopped on historical archive path binding:
retained guard bytes name their original run directories. A separate validation
reader verifies all 14 historical original/archive files and invokes unchanged
replay at the original paths. Frozen scorer/spec/parser and current v2 replay
remain unchanged. The failed attempt and repair provenance are preserved. This
operational repair changes neither the native outputs nor judgment rules.

**Early S3.1 host action software preparation passes 38 focused tests and 12
integration checks.** Verify/answer/stop proposals are bounded by source,
identity, branch, purpose, goal, policy/model hashes and budgets. Actual XOR tool
results have byte-bound receipts; invented values cannot authorize an answer.
Revocation invalidates cached current-epoch work. Restart checks the canonical
journal against an expected binding and caller-trusted head without reexecuting
tools. No external trusted-head store, crash/concurrent persistence, model
proposer, evolving subgoals or candidate learning is implemented by this helper.
S3.1 remains partial; its full S2.5/S1.1 dependencies are unchanged.

All 36 original file identities and acquisition receipt remain unchanged; the
nine-member seal rechecks before/after. All 1,060 prior artifact hashes still
match. **Zero downloads, training updates or new checkpoints.** Seven small
native evidence files total 142,505 bytes. Selected D0/D1/D2 and useful negative
exports remain; the untracked runtime draft remains byte-identical and excluded.
The [final suite](reports/validation_response_policy_v2_final_20261022.json) passes
**819 tests, one restricted-context MPS skip**. The pre-native suite remains
798 passed/one skip. All 33 stage IDs/dependencies remain unchanged. S1.4 remains partial with
abstention collapse; S1.3/S1.5 pending, S2.4 blocked, no model promoted and no
native job active.

The previous next-step proposal was a separate MPS model-to-host pilot. The
current architecture review supersedes that work ordering with E1/E2/E3; source
grounding, useful missing-information behavior and real receipts are integrated
acceptance checks in that build.
Do not modify this v2 arm after results. Existing B0/B1 evidence, retention,
calibration, public admission and budget/exposure gates still precede the fair
training study. No 20k campaign is admitted.

[Preregistered protocol](docs/experiments/20261022_Response_Policy_V2_Protocol.md),
[results and diagnosis](docs/experiments/20261022_Response_Policy_V2_Results.md),
[fixed scoring](reports/response_policy_v2_20261022/native_B1_scoring.json),
[raw semantic judgment](reports/response_policy_v2_20261022/judge_semantic_raw.json),
[reconciliation](reports/response_policy_v2_20261022/judge_reconciliation.json),
[consolidated evidence](reports/response_policy_v2_experiment_20261022.json),
[source preservation](reports/response_policy_source_reuse_20261022.json),
[host contract](docs/experiments/20261022_Host_Action_Contract.md).

## Preserved D2 qualification and storage package

22 October 2026, America/Chicago. Branch `codex/d2-capacity-runtime-pilot`. The user resumed this bounded D2 GPU package after the completed storage-maintenance pause. The package is complete; no native job remains active. No remote or push is configured.

**S0.5 actual 256/512-token D0/D2 capacity and S2.3 D2 tiny memorization/scope/restart now pass.** D2 generates all eight balanced training answers at updates 4 and 8, then stops by its declared rule. Independent saved-export evaluation confirms **train 8/8 and dev 14/16**, all EOS at the same cap 256. These are small reused synthetic fixtures. Useful architectural gain, broad transfer, retention and autonomous learning remain unqualified.

The original plain D0 update-8 learning diagnostic remains selected. D1 update-28 and its failed rate arm remain preserved comparison artifacts. No developmental or post-freeze model is promoted. The unfinished `src/replica_lab/runtime_pilot.py` draft is preserved separately and is not part of this package.

## Single original model and GPU execution

Reuse **`checkpoints/minicpm5_source`**, immutable MiniCPM5-2B revision `f97400052a43d642bbc6e9975e2397e3ae6a6b52`, with its existing sealed lock and tokenizer/template. There is one original source template for ongoing experiments. All 36 original files retain identical sizes, inodes and timestamps; the acquisition receipt is unchanged; all nine sealed inventory files reverify by hash. **Zero downloads** occurred. Missing/corrupt source fails offline instead of fetching automatically. [Final source check](reports/D2_source_reuse_final_20261022.json).

Mac Studio M2 Ultra: 24 CPU/60 GPU cores, 64 GiB unified memory, macOS 26.6.2. Native training and generation use **MPS GPU FP32, CPU fallback disabled**, one native process at a time. Tokenization, file-value audits and small contract tests use the host CPU. Python 3.12.14, torch 2.14.1 and Transformers 5.20.2 remain locked. The Neural Engine is not used by this backend. [Accelerator review](docs/experiments/20261022_Apple_Accelerators.md).

Every fresh capacity/tiny arm loads the same local source; within-pair continuation restores its own retained completed-update checkpoint. Selected D0/D1 states are comparison artifacts rather than substituted D2 parents. The completed historical 256 pair binds to actual Git objects at `9bc719b`; its receipts were preserved. New 512/tiny pairs use the unchanged storage-aware training hashes pinned in [the resumption plan](configs/d2_resume_plan_20261022.json).

## Measured D2 results

D2 trains **all 2,516,756,480 originals / 381 tensors**, including embeddings/head, plus **895,490 new parameters / 48 tensors**. No original is frozen. Added memory/workspace is 64-wide, four slots, two workspace steps and four heads at the final block. Experimental Muon 8-bit and FP32 auxiliary AdamW own every tensor once: 294/87 original and 16/32 new tensors. Original/new peak rates are 1e-4/1e-3, warmup two, constant schedule, zero decay, seed 17, microbatch one and accumulation eight.

| Gate | Updates and fresh-process restart | Sampled peak MPS live / driver GiB | Answer result |
| --- | --- | ---: | --- |
| Actual 256 tokens | 0→1→2, reused qualified pair | 26.751 / 29.072 | Capacity only |
| Actual 512 tokens | 0→1→2 | 26.876 / 30.080 | Capacity only |
| Tiny, actual 55 tokens | 0→4→8, declared maximum 64 | 26.653 / 29.195 | 8/8 at 4 and 8; saved train 8/8, dev 14/16 |

All 381 originals receive finite nonzero gradients and independently verified saved changes at each audited boundary. Added gradients open from two null output projections at update 1 to all 48 by update 2. All 48 sidecars change during each continuation. Runtime/export native and sidecar bytes match; all compared reload logits have maximum difference zero. Optimizer tensor state is 6,266,291,904 bytes. Both capacity and tiny chains pass strict metadata/optimizer/RNG/cursor/history/generation/guard validation with byte-bound native audits. Metadata mode does not independently rehash every large runtime payload.

All six training guards and both independent evaluation guards pass, with no observed swap growth. Declared limits are 40 GiB driver, sustained 512 MiB swap growth, host free minimum 15% over three samples; driver cadence .25 s, host cadence 2 s. Samples can miss transients; RSS/driver overlap. Native restart is verified, while full-size uninterrupted-versus-split numerical equivalence remains unmeasured.

Unique cost across all three evidence chains is 12 updates, 15,808 sequence tokens, 192 supervised tokens, 258.749 s update time and 634.273 s profiled trainer lifetimes. This resumption reuses the old 256 pair: **10 new updates**, 11,712 sequence tokens, 213.068 s update time and 500.531 s profiled trainer lifetimes. Independent saved-export evaluation adds 23.981 s. Audits/preflight/tests are additional work. [Source/workspace token accounting](reports/D2_token_accounting_20261022.json) reports added input views separately from native sequence tokens.

[Protocol](docs/experiments/20261022_D2_Protocol.md), [D2 results](docs/experiments/20261022_D2_Results.md), [consolidated evidence](reports/native_augmented_D2_experiment_20261022.json), [256 validation](reports/validation_D2_capacity256_20261022.json), [512 validation](reports/validation_D2_capacity512_20261022.json), [tiny validation](reports/validation_D2_tiny_20261022.json), [independent answer audit](reports/validation_D2_saved_answers_20261022.json).

## Storage and code checks

The [latest user-authorized retirement](docs/experiments/20261022_Checkpoint_Retirement_After_D2.md) removes **19 superseded payload directories and 10 orphan pool weights**, after the D2 experiment evidence was archived. Completed capacity probes, intermediate D1/qualification exports, D0 update 12 and D2 update 4 are retired. Their raw reports, answers, profiles and 100 byte-verified small metadata archives remain. Archived runtime audits describe the pre-retirement files and cannot be rerun against deleted payloads.

The original sealed source and acquisition receipt are unchanged. Selected D0 update-8, D1 update-28 and latest D2 tiny update-8 **exports and resumable checkpoints** remain, as do both useful final negative exports. All 35 protected byte claims reverify, and all protected/source/raw file identities remain unchanged. No model was loaded or downloaded for cleanup. [Plan](reports/checkpoint_retirement_plan_after_D2_20261022.json), [completion journal](reports/checkpoint_retirement_after_D2_20261022.json).

Observed free space increased **123.24 GiB**, from **465.76 to 589.00 GiB**. These are system-wide filesystem observations, not unique APFS extent accounting. The initial guard stopped on ctime-only drift in one unused pool file; its fresh full hash matched its content address, and a narrow recovery finished the remaining orphan removals without changing the original guards. The initial journal and recovery record are preserved.

Four pool versions remain, all referenced by retained native files. APFS sharing continues for identical future saves. End each completed experiment by archiving evidence, retaining chosen results/restart points and useful negatives, then explicitly retiring superseded payloads and orphan cache objects. Unique learned weights and optimizer states consume real space; deduplication alone cannot bound their growth. The [earlier cleanup](docs/experiments/20261022_Checkpoint_Cleanup_Applied.md) remains a historical record.

The D2 package's full suite passed **479 tests**, with one restricted-context MPS skip. The subsequent retirement/recovery and storage helpers pass **75 focused tests**, including real tiny-file APFS checks, with no failures/skips. [Retirement validation](reports/validation_checkpoint_retirement_after_D2_20261022.json), [D2 test evidence](reports/validation_D2_code_20261022.json). Training-bound implementation files stayed unchanged. Source, weights, runs and caches remain outside Git, with the artifact guard enabled.

## Next concrete gate

M0a remeasures the corrected MiniCPM5 inference capacity path under the new
50 GiB default with a fresh matching protocol/code pin
and evaluates the corrected action menus on fresh matched families. Reuse
existing regression coverage; preserve old capacity/ranking reports as
historical evidence.

M0b optionally compares Spark BF16 after native/resource qualification; it does
not block the MiniCPM5 sequence. Follow document 15 for memory-layer stages,
verified RSI and the MiniCPM-V visual-training path.

M1 then establishes fresh chronological train/dev/promotion/audit roles and
one typed compiler path using the existing ledger, catalog and publisher.
M2 builds a functional multi-chunk adaptive-memory trainer and one useful
memory path before M3 tests shared-depth coupling. Follow
[the detailed acceptance matrix](docs/architecture/15_Research_Crosscheck_and_Backbone_Plan.md)
and [loop instructions](docs/architecture/16_Loop_Agent_Instructions.md).

The exposed 2,000-target release remains infrastructure evidence.
[The S2.4 proposal](docs/experiments/20261022_S2_4_Screen_Proposal.md) still
requires eligible fresh data, equal information, matched parents, fair search
and parameter/compute controls. Thin early loop preparation may proceed;
selected-system longitudinal claims require the developmental comparison
and S2.5 selection. No 20k campaign or model promotion is admitted by this
documentation update.

Useful memory benefit, ten-cycle learning, authentic Replica fidelity,
native VLM, augmented KV reuse, ANE acceleration and Blackwell residency
remain separate open gates.

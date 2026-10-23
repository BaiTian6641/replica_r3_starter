# Execution state — E2/E3/C16/E4/E5 integrated build

**Latest verified package (23 October 2026 UTC).** Full suite **1125 passed /
21 subtests**. The connected E3 mixed-outcome family now has **4/4
head-conditioned and 4/4 no-head completions**, and the copied-case
fresh-process restart passes **12/12** acceptance checks. Earlier write_file and
set_constant negatives remain preserved; this is a bounded synthetic result, not
general capability. E2 retrieval/capacity, C16 identity contracts, E4 fast
memory/artifacts (delayed-use candidate rejected), and E5 procedure plus opt-in
reranker remain as recorded. No active native job. See the
[E2/E3/E4/E5 handoff](docs/experiments/20261023_Handoff_E2_E3_E4.md).

**Current work — E3 bounded mixed-outcome completion and restart (23 October
2026 UTC).** The shared width-256/eight-slot/two-iteration core, independent
memory/workspace readers, exact task journal, typed actions, real fixture tools,
independent success check and fresh-process restart are connected. On the mixed
family, **4/4 head-conditioned and 4/4 no-head episodes completed** with correct
repairs/passing evaluators, 0 malformed actions and 0 false success claims. The
action-kind head is rejected (no measured benefit). Fresh-process continuation
passes **12/12**: process 1 observed exit 1 and stopped; process 2 resumed the
same journal/ledger, repaired once, passed the evaluator and did not replay
completed actions. The success head remains **untrained/uncalibrated**; no
general capability, confidence or personal-fidelity claim. The sealed source
is unchanged and no native job is active.

**E2 reader candidate (23 October).** A D1 candidate trained `record_writer` and
both memory readers with answer loss on 24 source-separated cases. Loss fell
but dev+held-out exact accuracy remained **0/16** for parent and **0/16** for
candidate; the candidate emitted training values rather than copying held-out
values despite correct lexical retrieval. Decision: **defer**, no default
change. This rules out “no answer-loss training” as the sole cause and points to
reader/writer representation or supervision; the 10 held-out cases must not be
retuned. See
[reader candidate results](docs/experiments/20261023_E2_Reader_Candidate_Results.md).

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

**C16 identity interface (23 October).** Versioned source episodes, approved
conditional principle hypotheses, dated snapshots, mode-bound identity contexts,
parallel-branch state, correction/revocation handling, a null path and an
optional zero-initialized identity reader are implemented with fictional
contract tests. The runtime can include a bounded approved identity section in
its prompt when explicitly supplied; the default is byte-identical and identity
has no verifier or task-success authority. No authentic personal source,
identity training, native identity-port attachment or fidelity evaluation
exists. See [identity interface](docs/architecture/09_Replica_Identity_Interface.md).

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

The largest actually qualified capacity path is **16 units / 2,043 source tokens
/ 256 prompt tokens / 8 generated tokens**, plus a modules-only 16-unit backward.
All 12 schedule points completed; peak sampled driver memory **12.51 GiB**, zero
observed swap growth, 69% minimum free. D2 capacity was not declared or run.
See [E2 results](docs/experiments/20261023_E2_Retrieval_and_Capacity_Results.md).
Validation for this increment: full suite **1091 passed / 21 subtests**, E2
dataset tests 3 passed; receipt
[validation_e2_followup_20261023.json](reports/validation_e2_followup_20261023.json).

**Open:** held-out configuration repair, correct action selection/arguments,
full-core plan continuation, success-head training/calibration, novel-value
E2 writer/reader transfer, ranking quality and broad capacity beyond the
qualified 16-unit/256-token workload. These are
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

S1.2 supplies the bounded 2,000-target release. Query-policy v2 resolves termination while leaving abstention collapse and evidence-use failures; separately predeclare a grounded model-to-host pilot and complete independent evaluation/public admission. The [S2.4 proposal](docs/experiments/20261022_S2_4_Screen_Proposal.md) specifies equal information, parents, fair search and parameter/compute controls; numeric budgets and S1.5 admission remain unresolved. Existing diagnostics do not justify the 20k campaign or model promotion. Early host-loop preparation may proceed separately; selected-system integration still depends on S2.5.

Cognitive contracts, identity/lineage, source permissions and the proposed semi-cognitive loop remain as specified in [the design contracts](docs/experiments/20261022_Cognitive_Contracts.md). Useful memory benefit, ten-cycle learning, authentic Replica fidelity, VLM, augmented KV caching, ANE acceleration and Blackwell residency remain separate open gates.

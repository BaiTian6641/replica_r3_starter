# Execution state — integrated memory, cognition and learning


**M2/M3 verified packages (24 October 2026 UTC).** M2's shared-value SGDN-2
diagnostic passed 4/4 dev and 4/4 held-out after context removal, above linear
(1/4 and 2/4) and all null controls, with correction, restart and 2/2 retention.
The unique-value arm memorized training values and was rejected. M3 then passed
`accept_m3_ownership_interfaces`: shared and private both scored 8/8 dev+held-out,
with independent per-depth writers, a defined convex shared reducer, update-two
historical gradients, gradient-cut, authenticated shared/private images and two
fresh-process restarts. Shared/private answers agreed 8/8. Private uses 131072
state bytes versus shared 65536, so no capacity-matched superiority is claimed.
Peak driver was 11.0043 GiB and selected training took 215.896 s. See
[M2 results](docs/experiments/20261024_M2_Dynamic_Memory_Results.md),
[M3 results](docs/experiments/20261024_M3_Depth_Ownership_Results.md) and
[the verdict](reports/m3_depth_comparison_attempt02/comparison.json).

**Current blocker:** both M3 candidates score **0/2** on retention replay versus
2/2 for their parents (`copper` shared, `amber` private). The ownership
verdict does not accept this regression. Inference-only ablations are complete;
zero, disabled and full-context recover 2/2 in both arms. No single-port ablation
recovers or exactly reproduces the failure. Use fresh replay/audit splits before
any repair or M4 expansion. Full integration passed **1249 tests / 21 subtests
in 133.03 s**; later focused M3 regressions also pass. M0b Spark and authentic
identity fidelity remain unavailable optional tracks.

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
No new weights, adapter, memory mechanism or RSI capability were implemented.

**Next work:** fresh retention-replay training/development/audit roles and a
bounded repair candidate → M4 static lookup/dynamic routing if retention and
delayed-use both pass → M5 developmental comparison and bounded RSI.
M6 learned multi-rate levels → M7 native VLM. M0b is optional and does not block MiniCPM5. Identity
work proceeds only with permitted sources; calibration precedes control use.
D0/D1/D2 fairness and release admission remain required before
selected-system/broad-campaign claims.

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

# S2.4 one-seed screen proposal — 22 October 2026

**Status: blocked proposal, not a final preregistration or launch authorization.**
The [S2.4 dependency](../stages/03_Native_Architecture_Study.md#s24--run-one-seed-d0d1d2-and-matched-capacitycompute-screen)
requires S1.5 admission. Completed D0/D1/D2 diagnostics qualify bounded
infrastructure; their different rates, search histories and stopping times do
not constitute this comparison. No S2.4 GPU training starts until the
[S1 prerequisites](../stages/02_Data_and_Evaluation.md) and the unresolved items
below pass and are frozen in a versioned manifest.

The current work package prepares an S1.2 2,000-target, 1,000-world synthetic
release spanning four shared mechanisms and an S1.4 parser/control preflight.
Those mechanisms remain shared across roles: disjoint worlds test unseen-world
transfer, not unseen-mechanism transfer. Neither preparation alone admits the
20,000-target campaign. S1.3 public-source/independent-benchmark admission
remains pending. Exposed fixtures cannot become a sealed final audit.

## Comparison and information boundary

| Arm | Trainable scope and permitted input |
| --- | --- |
| B0 | Frozen original, task prompt without retrieved source-memory; capability reference with an intentionally different information condition |
| B1 | Frozen original, strong source-text memory and the same permitted tools/evidence as trained arms; primary evidence baseline |
| D0 | All 2,516,756,480 original parameters / 381 tensors, including embeddings/head; source text remains in the prompt |
| D1 | All 895,490 new parameters / 48 tensors; all original parameters frozen, with differentiable downstream paths preserved |
| D2 | All 381 original plus 48 new tensors; full augmented tuning without freezing original layers |
| Feed-forward control | Implement and qualify a replacement with matched added parameter capacity and measured compute before attributing a gain to recurrent workspace steps |

All trained arms start fresh from the **same single local original template**,
`checkpoints/minicpm5_source`, sealed MiniCPM5-2B revision
`f97400052a43d642bbc6e9975e2397e3ae6a6b52`. Retained diagnostic winners are
historical comparison artifacts, not substituted parents. Use identical
eligible source records, native prompt text/order, targets, tools and example
plans across trained arms. Keep source text in every trained prompt; augmented
branches encode that same evidence rather than receiving privileged facts.

Start from the qualified final-block, 64-wide, four-slot, two-step configuration.
The feed-forward implementation, parameter tolerance and compute matching rule
are unresolved. Missing/shuffled evidence and memory/workspace disable tests
must be specified. Disabling a branch only at inference measures sensitivity;
retraining without it is a distinct condition and consumes training budget.

## Budgets and measurement

Propose one initial seed and **at most two predeclared search configurations per
trained arm**, with identical selection opportunities and no unregistered extra
search after a poor result. D2's original/new rates require explicit joint
configurations; an unrestricted Cartesian search would violate this cap.
Count warmup, interface preparation, rejected/failed searches, replay,
verification and evaluation in costs. Existing diagnostic search costs remain
historical costs, not evidence of equal campaign budgets.

Freeze numeric rates, schedules, total update/token limits, checkpoint-selection
rule and common generation settings after release tokenization and bounded
resource accounting, before any comparison training. A two-update capacity
pass does not establish sustained-run feasibility. Do not use tiny training
perfection as the useful-gain stopping rule.

Report native sequence tokens, supervised answer/EOS tokens, support-writer
tokens, workspace inputs, replay and generated tokens separately. Equal native
tokens or update counts do not imply equal total architecture compute. Preserve
an equal-token comparison and a separately defined **measured total-compute**
comparison, including training and inference time/overhead. Freeze cache mode,
decode cap, prompt/template/thinking settings, tool budget and accounting
boundaries; disclose any unmatched control instead of labelling it matched.

## Evaluation, exposure and decisions

Split world/source families before derivatives; paired comparisons and
uncertainty use the independent world as the unit, not its related queries or
paraphrases. Preserve train, development, calibration, reward, promotion and
audit exposure boundaries. Development supports rate/checkpoint selection;
calibration has separate cases. Cap and log independent promotion queries.
Final audit cannot guide teachers, training, fixes or selection. Any inspected
audit fixture is exposed development/contract material thereafter.

Lock strict whole-response parsers, solver correctness, action validity,
format failures, constant-answer rejection, EOS/caps, evidence controls and
retention families before scoring candidates. Generated answers, not declining
NLL alone, determine task results. A four-mechanism synthetic release does not
measure authentic Replica fidelity or broad external retention.

The roadmap's **+2 percentage-point useful gain, at most 1-point mean retention
loss and 2-point critical-family loss** remain provisional margins. Sample
sizes, clustered interval/test rules, multiplicity/exposure handling and the
power check are unresolved. A small or zero-difference sample cannot certify
these limits. Hard causality, scope and identity invariants must all pass;
average gains cannot offset a hard failure. A one-seed screen shortlists or
rejects mechanisms; it does not replace S2.5's independent families/three seeds
or promote a developmental model. Tie, negative and inconclusive are valid.

## Execution and checkpoint retention

Use serial M2 Ultra **MPS GPU FP32**, experimental Muon 8-bit plus eligible
auxiliary FP32 AdamW groups, nonreentrant checkpointing and
`PYTORCH_ENABLE_MPS_FALLBACK=0`. One native model job runs at a time. Keep source
verification/acquisition offline and reuse the sealed bytes; missing/corrupt
source fails without fetching. CPU handles host I/O, solvers and file audits.
This backend does not execute on the Neural Engine.

Retain the existing 40 GiB sampled driver limit and sustained host-pressure
stops, and freeze disk/headroom and run-time caps before launch. Stop on scope,
parity, nonfinite-update or resource failure; archive the negative instead of
silently freezing layers or extending budgets.

For new screen outputs, save only a terminal checkpoint and the restart state
actually needed by the declared process boundary. Archive reports, raw answers,
profiles, guards, receipts, manifests and saved-weight audits before retiring
designated intermediates. Use verified APFS sharing for byte-identical native
weights; different learned snapshots and optimizer states still consume space.
Pool orphan collection must follow an explicit reference inventory and the
retention ledger. Preserve the one original and existing selected results;
this proposal authorizes no deletion of unknown artifacts. The separate
untracked host-runtime draft remains outside this package.

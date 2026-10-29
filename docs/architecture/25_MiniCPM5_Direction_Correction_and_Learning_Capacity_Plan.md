# MiniCPM5 — direction correction and native learning-capacity program

**Direction reconciliation — 28 October 2026:** the active Spark-first text
program is [chapter 28](28_Spark_Persistent_Learner_and_Tiered_Memory.md), with
[revised worker instructions](16_Loop_Agent_Instructions.md). This chapter
preserves the dated v5.1/MiniCPM design lineage and its evidence. Native
multimodal training is deferred; MiniCPM is not a prerequisite for Spark.
Expanded P profiles and tiered storage may be engineered now, while trained
capacity promotion requires useful readout, correction and retention.
The v1 CPU pilot's 0/12 result uses an untrained attention-free stub and does
not establish a Spark full-context defect. See the
[current review](../../reports/spark_persistent_learner_review_20261028/README.md).
No historical result, source identity, sealed protocol or native gate is
changed by this direction note.

**27 October 2026 · Prospective experiment design. No new execution admission.**

MiniCPM5 is the mechanism platform for the [Spark v5.1 product direction](24_SparkX_2_6_v5_1_Refinement_Plan.md). Its original backbone contains **2,516,756,480 parameters**, 42 layers and width 2048. Spark has 36 layers and width 2560 with different attention geometry. Transfer the learning mechanism and data recipe, not tensor shapes or a MiniCPM checkpoint.

The immediate objective is to turn correct wiring and state continuation into useful native learning. Preserve current evidence: exact-bank readout works on its bounded fixture; native learned retrieval remains rejected; P train-fit generation remains 0/28; the vision diagnostic supports first-step overshoot, not visual competence. Increasing memory capacity is a staged experiment after a small useful learner exists.

## 1. Experimental order and stop points

The identifiers below are new **direction-correction studies**. They do not rename NL0–NL7, V0–V4, existing jobs or their failed receipts. A study becomes runnable only after its own protocol, source pins, budgets, trainable groups and evaluation roles are sealed.

| Study | Primary hypothesis | Control / output | Dependency |
|---|---|---|---|
| DC-V1 | A bounded optimization change addresses the observed first-step vision overshoot | Same encoder/data/projector geometry; null and smaller-update controls; train-only loss and free generation | Preserve the 27 October diagnosis; declare a new run |
| DC-R1 | Native record readout can use correct selected content with abstention | Exact context, exact selected-record, compressed codec and wrong-value controls | Use new role-sealed fixtures; do not reopen old held-out labels |
| DC-R2 | Native hybrid retrieval supplies useful records at adequate coverage | Learned dense vs dense+lexical+entity; oracle diagnostic separate | DC-R1 readout signal |
| DC-P1 | A bounded response-training recipe produces useful P-only output | Current single-step recipe vs one independently changed loss/rate/schedule | New protocol; no retrofit of rejected budget02 |
| DC-P2 | More stored capacity improves learning/retention at bounded active work | P16/P32, then P64; active2 fixed; width and routing controls | DC-P1 positive new-instance signal |
| DC-M1 | Sparse parameter values offer efficient factual learning | Mkv-4k smoke → Mkv-64k; raw-frequency vs background-specific writes | Native disabled parity, sparse optimizer contracts and useful response readout |
| DC-N1 | Offline preparation improves future learning | Ordinary initialization/support fitting vs first-order outer training | Useful inner learner |
| DC-N2 | Multiple clocks improve retention beyond added parameters | One vs two clocks at identical P capacity and teaching budget | DC-N1 or a fixed useful initialization |
| DC-N3 | Conditioned-teacher distillation or learned update rates improves the tradeoff | Verified CE vs forward-KL self-distillation; separately exact/truncated update gradients | Teacher competence; complete work accounting |
| DC-F1 | Fast association adds measurable benefit | E+P vs E+P+linear F vs gated/nonlinear F | Useful E/P; otherwise defer |
| DC-AV1 | Already-trained modalities support new learned conventions | Single-modality, transcript-only, mismatched-time and source-free controls | Full offline modality training passes |
| DC-J1 | E/P/vocab cooperate under turnover and interference | Eight M×L×V arms plus parameter-table attribution | Useful independent branches |
| DC-S1 | Chosen mechanism transfers to Spark | Spark-native disabled parity, small fit and fresh-process continuation | Mini mechanisms pass; Spark capacity preflight |

Choose one primary change per first diagnostic. Combining an encoder swap, dense reads, moment reset, projection, new loss and more steps in one run cannot identify the cause of success or failure.

## 2. Vision and audio: full training comes first

### 2.1 Repair the current vision learning path

Use the current base encoder/projector to isolate optimizer behavior. The latest result has finite nonzero gradients in all six projector tensors but an update norm around 1.019 and worse train losses after step one. A new bounded hypothesis should inspect effective update magnitude, gradient clipping, learning rate and initialization/scale while preserving the dataset and response contract.

Require both loss progress and meaningful output progress; a gold-rank change alone is not qualification. Compare correct pixels with sibling, shuffled and null pixels. If optimizer checks pass but a frozen-backbone projector cannot solve the fixture, declare a small full-co-adaptation control; do not require an impossible projector-only task as the universal gateway to full fine-tuning. Keep its result and trainable scope separate.

Only then compare the larger So400m NaFlex candidate. Mini's proposed projector is **1152→1024→2048**; Spark's is 1152→1024→2560. A zero-output/disabled bridge must preserve native baseline behavior. Seal encoder, processor, grid handling, projector inventory and masks before training. Processor default is 256 patches at the reviewed revision; ≤1024 patches is an explicit experiment.

### 2.2 Progressive multimodal curriculum

F0 checks native graph/gradients and loss accounting. F1 aligns modality interfaces. F2 fully fine-tunes the declared backbone, sensory encoders and interfaces, progressing through image, video, audio and paired AV. Maintain a text-retention stream at every stage. Test short inputs before long video/audio. F2 contains **no online-memory optimization**; it teaches the model to see, listen and reason.

F3 adds native memory read/write tasks and branch-absence training while retaining F2 abilities. F4 trains chronological learning episodes. Full offline training may co-adapt the stable backbone in these stages; deployed inner updates may not. Record trainable groups for every phase so “frozen backbone” never ambiguously describes both development and deployment.

For audio, the first reference is complete clips through SPEAR with native acoustic features; a transcript is only a control. Compare 10/12.5/25 model-facing tokens/s with ~50 Hz encoder features retained internally. Evaluate music/rhythm, brief environmental sounds, overlapping speech, prosody and multilingual slices separately. Noncausal windows cannot emit evidence before their end. No speech/music generation is required by this input-capability study.

## 3. Diagnose memory use before expanding it

The exact-bank experiment already passed a bounded selected-record readout endpoint. Freeze it as a control; do not train on its final held-out examples. It supplies no evidence that the model itself selects the correct record or reconstructs an exact literal from compressed vectors.

DC-R1 compares a source-in-context answer, an exact selected literal, a native compressed record, shuffled values and no memory under the same output grammar and adequate generation cap. Report complete-fact correctness and per-layer behavior separately. Accepting one correct layer while another abstains is not a complete multi-layer recall success.

DC-R2 replaces the oracle selector with a native query head, then compares dense, lexical and entity channels. The host enforces scope and source status only. Measure R@1/R@16, eligible-source coverage, conditional readout accuracy, accepted-correct coverage, precision, false recall and calibration. Evaluate retrieval failure and value failure separately; a threshold sweep cannot repair missing value representations.

Adopt E2 first at the Mini capacity target: 4096 compact records, 256 hot records, 4096 entity anchors. This is 24.5 MiB of declared payload, plus explicit variable payload/index/cache quotas. Compare compact-only and hot sensory evidence on tasks requiring detail or timing. Use at most sixteen selected records and the declared 200-packet read budget. Exact literal recall remains a distinct endpoint from parameter acquisition.

## 4. Persistent learning: loss, routing and capacity

### 4.1 Make P useful at a small scale

The current response diagnostic improved average NLL but had 0/28 teacher-forced exact and 0/28 greedy answers. Therefore, do not label the problem “only decoding.” Examine supervised response alignment, selected-expert signal, bridge magnitude, exposure during readout and target coverage under a new train-only diagnostic.

Use one or a few experts to establish learnability, then the historical P16 bank. Compare native routing with a separately labelled oracle-route diagnostic; an oracle-route pass cannot qualify product routing. Hold target strings/world IDs out of route features. A dense16 read control can diagnose sparse-access failure but must be charged at its actual cost.

Primary positive evidence is a gain on fresh paraphrases and new instances after teaching, relative to reset-P and identical no-update controls. Exact repeated training-answer fit is necessary for diagnosis but not enough. Free generation, correction and unaffected-skill retention must be reported alongside CE and token accuracy.

### 4.2 Capacity matrix

| Arm | Stored capacity | Active/read/write policy | Question |
|---|---|---|---|
| P16 | 12.58M weights | Initially two active; selected experts only | Historical mechanism control |
| P32 | 25.17M weights | Same active2 and response budget | Does more storage reduce interference? |
| P64 | 50.33M weights | Same active2; active4 separate | Is routing or storage the next bottleneck? |
| Wide P16 | 50.33M weights at doubled widths | Recount projections/active compute | Is representation width more valuable than expert count? |
| Mkv-4k | 4096×256 value weights | Top8; scoped sparse row updates | Does the native table work at all? |
| Mkv-64k | 65536×256 value weights | Top8; background-specific writes | Does granular parameter storage improve factual retention? |
| P+Mkv | Separate procedural and associative banks | Explicitly bounded access to both | Is combination better than capacity/compute-matched alternatives? |

Native P interfaces on Mini start at 2048↔256 and existing model-specific ports 13/27, with 41 retained as an observer/control where applicable. More port sites require a separately trained density arm. The Spark nine-site MAC estimates are **not** Mini measurements. Mkv initially shares a table across the two active Mini ports, with depth identity; it is not a fresh table at every layer.

Mkv's fixed keys and query projections must be trained to align teaching and later question access. Log overlap between write accesses and future native query accesses as a diagnostic, without using future query IDs to choose writes. A larger table that is never addressed correctly is not added useful knowledge capacity. Track collisions, utilization, dead slots, access entropy and interference by source family.

Keep 512 MiB as the proposed Mini P32 dynamic-state envelope plus 128 MiB replay. Adding Mkv-64k brings parameter/moment payload close to that envelope before other state; use a separate 768 MiB candidate envelope for the combined Mini arm, with measurement required. Current process memory/swap guards remain unchanged. Record dense-gradient fallbacks, allocator overhead and staging; sparse semantics do not guarantee sparse allocation.

No Mkv experiment starts before the row-sparse contract is declared and checked. The protocol must keep inactive rows immutable under momentum and weight decay, preserve optimizer state for selected rows across restart, aggregate duplicate-index gradients deterministically, define per-row step clocks, and measure actual allocation and staging. A zeroed dense gradient does not satisfy these conditions; until a row-sparse optimizer is qualified, full-table gradients and temporaries are charged. The 4,096-row smoke precedes 65,536 rows, and both wait for a useful small learner (DC-P1); they do not replace it.

### 4.3 Data stream design

Use train/dev/audit splits by independent world, source document, entity family, audio clip/speaker and video scene. The test task must be unknown before teaching so that ordinary pretrained knowledge cannot masquerade as learning. Include new factual associations, compositional rules, exceptions, conflicting versions, unknowns, negative teaching and delayed correction.

Maintain separate categories: exact literals, semantic relations, procedures, visual conventions and acoustic/AV conventions. Derived augmentations stay in their source's split. Avoid random-ID shortcuts, answer-bearing route labels or globally unique prompt formats. The final test is sealed before selection; commit-time probes come from a development stream and are not called held-out proof.

The existing [dataset/curriculum registry](../08_Dataset_Acquisition_and_Curriculum.md) supplies broader modality and retention material. Dataset presence on Hugging Face does not settle media rights, license compatibility or leakage. This review changes the design, not dataset acquisition status.

## 5. Stronger nested learning, with an explicit credit path

Use two stages of evidence. First, useful inner learning must work with a fixed initialization. Second, demonstrate that outer training improves post-update behavior at the same support/update budget.

Before any later stage runs, the preceding declaration must be sealed. The authoritative order in the [v5.1 design configuration](../../configs/sparkx_2_6_v5_1_direction_design.json) is:

| Order | Stage | Required declaration before the run |
|---:|---|---|
| 1 | Small useful inner learner | Inner protocol, trainable groups, step count, replay policy and post-teaching new-instance endpoint |
| 2 | Teacher competence and work | Teacher-competence check plus all teacher-forward positions, student rollouts, replay passes, evaluation positions and target-checking work |
| 3 | Gradient approximation | First-order identity Jacobian, exact short unroll or truncated unroll, and the derivative path each omits |
| 4 | Outer loop | Support/future-query split, source-free fraction and retention endpoint |
| 5 | Multiple clocks | Single-clock control at equal total parameters and work; clock objectives, transition rules and slow-pool population |
| 6 | Self-distillation | Teacher competence passed, every teacher/rollout/replay/evaluation pass declared, forward-KL primary and reverse-KL separate |
| 7 | RSI / learning-policy improvement | Qualified single- and multi-clock evidence; independent episodes at equal cost; no modification of verifier, source authority, labels or promotion criteria |

Teacher competence, gradient approximation and the complete teacher/rollout/replay/evaluation account are therefore declared before the outer loop, multiple clocks, self-distillation or RSI. Personality remains a separately supplied and separately evaluated input; it is not inferred from these experiments.

Each outer episode contains support teaching, future source-free queries, older-skill probes and grounding-retention tasks. Interleave corrections and at least 32 eligible teaching events so a 16-event slow clock is exercised twice. Held-out future queries never contribute to inner updates. During offline training, query losses can train stable readers/router/backbone as declared; during deployment they remain frozen.

Compare ordinary training against first-order initialization learning. Then test functional short-unroll updates for three matrix-specific rates per expert. A first-order identity-Jacobian estimator omits the rate/optimizer path; it cannot be cited as evidence of learned step sizes. Exact short unroll, truncated unroll and an amortized learned writer are separate mechanisms, with numerical checks before native quality claims.

Test one and two clocks at equal total parameters, active experts, support data and charged work. The slow pool receives recent plus age-balanced verified replay; it is not automatically populated by copying fast weights. Evaluate whether slow knowledge survives fast-state reset and whether consolidation repairs or amplifies errors.

Conditioned-teacher forward-KL distillation follows a teacher-competence check. Charge student rollout, teacher forward, target checking, replay, anchors and gradient passes. Compare to verified CE at equal total wall time and positions, not just equal optimizer steps. Preserve pre-update teacher identity within a transaction. Test a frozen snapshot and dynamic-state EMA separately; no hidden second full backbone is assumed.

Optimizer-state reset is an ablation, not an already-established fix. Compare carried Adam state with event resets under declared clocks. A reset-m/carry-v variant requires two bias-correction clocks and mid-event serialization. The current NativeFusedAdamW policy remains unchanged until a separate optimizer protocol is admitted.

## 6. Joint E / parameter / vocabulary validation

Retain the existing NL6 factor meanings. In DC-J1, `M` denotes episodic/associative memory, `L` the qualified persistent learning branch and `V` the vocab overlay. Run all eight on/off combinations from a common branch-absence-trained parent. If Mkv is selected, add `P-only`, `Mkv-only` and `P+Mkv` attribution; do not hide table contribution inside an unlabelled learning toggle.

Teach the same eligible observations in every arm, then:

1. Remove teaching prompts and source KV; ask a new question.
2. Replace phrase slots with unrelated phrases and query the original concept via base tokens.
3. Restart, reactivate the concept under a different slot/name and ask a new instance.
4. Introduce interference, a corrected rule, unknowns and a revoked source.
5. Remove E/F/workspace/raw media/answer-bearing metadata and test P or Mkv weights alone.
6. Reset or substitute those parameters and verify that the claimed gain disappears or decreases.

Measure surface accuracy and total decoding time across vocab arms; raw token accuracy/perplexity is not comparable across segmentations. Confirm base pre-softmax logits remain unchanged by phrase-row permutations and that no stale slot is used as a durable memory key. Atomic phrase positions affect RoPE/local-window geometry and require temporal-task regression checks.

For AV learning, use unfamiliar instances of an already-perceivable convention: a taught sound sequence means a particular visible device state, for example. Hold out speaker, timbre, scene and timing configurations according to which invariances are being claimed. Keep waveforms while removing transcripts, then do the reverse. Time shifts must change the expected relationship when the task requires synchrony.

## 7. Decisions, measurements and promotion

Prospective targets are development proposals, not changes to completed gates:

| Dimension | Proposed decision rule |
|---|---|
| Useful parameter learning | ≥10 percentage-point new-instance gain over the strongest matched nonlearning control; paired source-group 95% CI excludes zero |
| Retention | ≤2 percentage-point drop on predeclared general and modality score families; publish each family |
| Capacity expansion | Learning/retention improvement at fixed active work, or lower total cost at matched quality; otherwise retain the smaller profile |
| Local-loss shortcut | Within 2 percentage points of qualified behavioral learning and a measured total-cost improvement |
| Vocab | No identity corruption; reactivation within 2 percentage points of resident-row control; correct fallback |
| Restart | Read-only parity and subsequent-learning parity separately; weights, optimizer and clocks restore |
| Temporal/native sensory use | Relevant pixel/audio/time perturbations cause the appropriate behavioral change |
| Long run | Bounded live bytes, replay/journal growth, latency and queue; no stale-scope or double-update failures |

Use at least three independent seeds and balanced task orders for qualification, paired by source/world. Determine audit sample size from a development power/precision calculation. Three generated questions per teaching event are not independent sources, and reusing an audit to select transactions destroys its role. Report negative results and all predeclared primary comparisons.

Collect physical forwards by phase, computed text/visual/audio positions, encoder calls/cache hits, generated positions, backward calls, optimizer steps, selected expert/row counts, actual bytes, wall time and queue latency. The corrected historical P teaching total is 3429 processed positions, not 2670; missing target-production passes must not recur. Current historical generation/accounting receipts remain unchanged.

## 8. Handoff to Spark

Transfer only mechanisms with a useful Mini result. Seal Spark's actual 2560-width interfaces, original GELU blocks, GQA geometry, sliding/full masks and tied output head. Repeat disabled-extension parity, short learning fit, native retrieval, P-only/Mkv-only intervention, restart and cost measurements on Spark before scaling full training.

The Spark product candidate targets P64 and a larger E2 sensory store; Mkv remains an independently qualified option. A single Mini pass cannot establish full Spark convergence, 24-hour endurance or lifelong learning. The user-supplied personality description remains an explicit later input; it is not inferred from research papers or affect labels.

For detailed architectural equations, optimizer alternatives, source qualifications and arithmetic, use [the reviewed v5.1 plan](24_SparkX_2_6_v5_1_Refinement_Plan.md) and [review report](../../reports/sparkx_v5_1_review_20261027/README.md).

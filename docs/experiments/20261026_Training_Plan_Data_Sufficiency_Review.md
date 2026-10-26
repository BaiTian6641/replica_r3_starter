# Data needed for a convincing MiniCPM / SparkX result

26 October 2026. Research review and prospective budget. No model run or corpus acquisition was performed for this review.

**Budget approximately 2B processed positions for a convincing MiniCPM research attempt, then 5B for SparkX.** A concrete starting recipe is **1.2M ordinary training records plus 6,000 learning episodes for MiniCPM**, and **3M records plus 15,000 episodes for Spark**. These are estimates to test with scaling curves, not measured convergence thresholds. The practical uncertainty ranges are **1–3B and 3–8B positions**, respectively. Both retain the proposed full architecture and full joint fine-tuning scope.

Here, “convincing” means reproducible multimodal competence, causal benefit from the proposed learning mechanisms, persistence after restart, and retained old skills on independent data. It does not imply broad product readiness or recursive self-improvement. No published result establishes a sufficient dataset size for this exact combination of modules.

The [calculator output](../../reports/convincing_result_data_budget_20261026/estimate.json) distinguishes data, repeated processing, outer updates, inner updates, and evaluation work. The [assumptions](../../configs/convincing_result_data_budget_v1.json) are editable planning inputs. Exact figures below are arithmetic consequences of those assumptions; their precision does not imply equal confidence in the forecast.

## What the repository evidence supports

The review covers the consolidated architecture revision 4.5, acquisition/curriculum documents, intrinsic-learning and nested-learning protocols, current acceptance rules, native timing evidence, and the mixed-precision cost review. The [input manifest](../../reports/convincing_result_data_budget_20261026/input_manifest.json) pins 15 relevant files and the observed Git revision.

| Evidence | Interpretation for data planning |
|---|---|
| Proposed 10k visual examples, 15 audio hours, 500 AV episodes and 1k learning episodes | A candidate pilot pool. It is not the final training recipe, and it cannot all fit into a 2M-position diagnostic run. |
| Seven real images from six source groups, four train / one development / two audit | Acquired and independently verified as an evidence tranche. The [review receipt](../../reports/real_image_tranche_20261026/independent_review.json) still says `training_admitted=false`; this is not a trained multimodal corpus. |
| [Balanced vision diagnostic](20261026_Vision_Balanced_Alignment.md): 3/14 train-fit, 2/18 development, caption-following accuracy zero | Diagnose the failed learning behavior before scaling. This projector-only result does not determine whether full FT will succeed. |
| Newer [vision diagnostic](20261026_Vision_Collapse_Diagnostic.md): correct response NLL 1.52 versus fresh 10.27, but shuffled first-token accuracy 7/26 versus correct 5/26 | Target-text fitting occurs. Robust image-dependent generation remains unestablished; this strengthens the case for diagnosing shortcut use before adding data. |
| [Selected-record memory readout](20261026_Memory_Bank_Value.md): literal values 124/128, native exact answers 19/152 | The readout can fail despite useful stored information. More stored examples alone does not repair that failure. |
| [NL0 qualification](../../reports/nested_learning_nl0_20261026/qualification.json): 28 software/numerical checks passed | Contracts are qualified on standalone fixtures. Native source-free parameter learning, multimodal learning and policy improvement remain unqualified. |
| Sustained MiniCPM timing: 8.4216 seconds/update, 0.11874 updates/s, 52.25 processed positions/s | Measured on 440 positions/update, FP32 MPS and the existing backbone. It measures throughput, not sample efficiency or convergence of the complete proposal. |

The latest receipts supersede older statements that *no* real media or NL0 implementation exists. They also show that visual target-text fitting is possible, while the desired dependence on the correct image remains unestablished. Concurrent work outside these pinned inputs is not treated as a completed result.

## What the literature changes

| Primary research | Relevant evidence | Implication for this plan |
|---|---|---|
| [LLaVA-1.5](https://static.hliu.cc/files/llava/improved_llava.pdf), Tables 3 and 7 | 558k alignment examples and 665k instruction examples, starting with pretrained components | Hundreds of thousands of well-chosen examples can establish useful image-language behavior. Its image-focused recipe does not cover our audio, temporal memory or plastic-learning objectives. |
| [Molmo2](https://arxiv.org/html/2601.10611v1), §3.2 and Tables 12/14 | All-parameter pretraining: 32k steps, batch 128, maximum length 2,560; SFT: 30k steps, batch 128, maximum length 16,384. The 4B model reports approximately 490 + 7,500 + 3,200 H100 GPU-hours across its three stages | A much stronger grounding/video reference than a caption-only recipe. Its vision encoder accounts for substantial work. Our smaller data budget is a focused research attempt, not a forecast of Molmo2-level breadth. |
| [SALMONN](https://proceedings.iclr.cc/paper_files/paper/2024/file/476ab8f369e489c04187ba84f68cfa68-Paper-Conference.pdf), §4.1–4.2 | Alignment includes 960h LibriSpeech, 1,000h GigaSpeech and 2,800h WavCaps; instruction tuning reports about 2.3M examples / 4,400h. Only about 33M Q-Former/LoRA parameters are updated | Fifteen audio hours is an engineering pilot. Audio competence needs diverse speech, sound and music evidence. The reported task-hours/rows reuse recordings and must not be summed as independent media. |
| [SmolVLM](https://arxiv.org/html/2504.05299v1), §3–4 | The 2.2B family demonstrates small-model multimodal capability. Data balance, structured media markers, prompt masking and excessive rationale supervision materially affect performance | Test concise grounded targets and text-retention balance on development data. Its 14% text / 33% video sampling mix is not directly comparable to our position-based mixture, and its CoT result is not a universal optimum for MiniCPM. |
| [Nested Learning / HOPE](https://arxiv.org/html/2512.24695v1), §9.3 | Language-model experiments train approximately 760M parameters on 30B tokens and 1.3B on 100B, from scratch | This supports testing nested mechanisms, but supplies no evidence that adding our modules to a pretrained decoder will converge in a few million positions. It also does not imply that we must repeat from-scratch pretraining. |
| [LIMA](https://arxiv.org/abs/2305.11206) | 1,000 curated instruction examples on a pretrained 65B text model | Useful evidence about eliciting existing abilities. It is not evidence that 1,000 examples teach a new audio path, grounded memory or persistent parameter transfer. |

For scale comparison only, Molmo2's first two `steps × batch × maximum length` products are **10.49B and 62.91B position slots**. These are maximum-capacity products, not measured non-padding tokens or independent examples. Its approximately **11.2k H100 GPU-hours** includes a long-context stage absent from our short-context throughput model. [Molmo2 training details](https://arxiv.org/html/2601.10611v1#S3.SS2).

The recommendation is an engineering inference from these precedents, pretrained components, and the project's narrower first demonstration. There is no justified “tokens per parameter” conversion that makes the result certain. Spark does not automatically need twice the data because its backbone is larger; the higher proposed budget buys coverage and repeated integrated experiments.

## Review findings and plan changes

| Priority | Finding | Recommended change |
|---|---|---|
| P1 | The current learning failures could be objective, labeling, masking, optimization or routing failures | First obtain balanced tiny-set fit and correct sensory counterfactual behavior; diagnose failure before increasing data. Preserve existing gates. |
| P1 | Public captions/QA do not exercise correction, lexical turnover, source removal and parameter transfer | Allocate an explicit chronological episode corpus. The revised minimum is 32 eligible teaching transactions, not the older 4–16-observation suggestion. |
| P1 | A dataset catalog and a downloaded asset are different from an admitted training example | Retain source-group splits, decoded-media QA, deduplication and benchmark exclusions. Count independent assets separately from derived questions. |
| P1 | A single final run cannot establish the architecture's contribution | Fund seeds, ordinary-preparation and one-clock controls, the M/V/L factorial, recurrent-core controls, state-reset interventions and retention evaluation. |
| P1 | Long video / high-resolution encoder work can exceed the earlier fixed extension allowance | Profile actual raw frames, crops, encoder positions, audio seconds and every live re-encode before committing a formal GPU-hour estimate. |
| P2 | Aggregate loss and the provisional 30/20/20/15/15 mix can hide a failed component | Keep per-objective losses and held-out metrics. Compare text shares and concise/long supervision at matched work; do not adopt a published ratio without matching its units. |
| P2 | Two hundred development groups were being asked to support small-effect conclusions | Keep them for screening. Design the sealed evaluation around independent groups and the intended effect size. |

These changes refine the research sequence. F2/F3 remain full offline training of backbone, sensory encoders, bridges and intrinsic modules. F1 warmups and diagnostic ablations retain their explicitly narrower scope; they do not replace the final full scope. F4 uses bounded inner updates inside an explicitly trained outer procedure. F5 consolidates and tests the integrated model. F6 policy evolution needs separate fresh-task evidence and has no defensible fixed data requirement yet.

## A concrete data recipe

Counts below are *distinct curated training-record targets after deduplication*, before repeating examples. Several records may refer to one media asset. Evaluation data are additional and are never persistent training feedback. Mean positions include prompt, media representations, answer and executed padding; raw encoder patches are accounted separately as compute.

| Training bucket | MiniCPM records | Spark records | Assumed mean positions/record | MiniCPM processed positions | Spark processed positions |
|---|---:|---:|---:|---:|---:|
| Text retention: instruction, reasoning, mathematics, code | 500,000 | 1,250,000 | 768 | 576.0M | 1,440.0M |
| Image: caption, region/count, OCR/document/chart, visual evidence | 250,000 | 625,000 | 1,024 | 384.0M | 960.0M |
| Audio: speech, non-speech events, music structure, perceived affect | 350,000 | 875,000 | 768 | 403.2M | 1,008.0M |
| Temporal / AV: tracks, order, sync, joint evidence, interference | 100,000 | 250,000 | 2,048 | 307.2M | 768.0M |
| Native learning episodes, separately counted | 6,000 episodes | 15,000 episodes | 33,792/episode | 304.128M | 760.320M |
| **Total** | **1.2M ordinary records + 6k episodes** | **3M ordinary records + 15k episodes** | **1.5 average exposures** | **1.974528B** | **4.936320B** |

The resulting position mixture is approximately **29.2% text / 19.4% image / 20.4% audio / 15.6% temporal-AV / 15.4% learning**. The 1.5 multiplier means weighted revisits averaging 1.5 exposures, including planned consolidation. It is not an instruction to run every source for an arbitrary fixed number of epochs. Evaluate after one exposure and continue selected data only when development results justify it.

The **1–3B MiniCPM range** corresponds roughly to **0.6–1.8M ordinary records and 3k–9k episodes** at this recipe. The **3–8B Spark range** corresponds roughly to **1.8–4.8M records and 9k–24k episodes**. Longer inputs, different repetition, or more replay change positions without increasing distinct data. A model can meet some gates below these volumes or fail above them.

At an illustrative two questions per asset, the central recipe requires approximately:

| Independent media target, subject to actual grouping | MiniCPM | Spark |
|---|---:|---:|
| Images | 125,000 | 312,500 |
| Audio clips at a 10-second mean | 175,000 / **486 hours** | 437,500 / **1,215 hours** |
| Temporal/AV clips at a 20-second mean | 50,000 / **278 hours** | 125,000 / **694 hours** |
| Eligible teaching transactions within new learning episodes | 192,000 | 480,000 |

These hours are duration arithmetic, not sufficiency thresholds. Music, speech, longer video and episode assets can substantially change them. Shared recordings across audio and AV must be grouped globally and counted once in the media union. Extra captions over the same image do not supply new visual diversity. Music fundamentals from isolated notes need separate checks on actual compositions/performances.

If only 50–80% of candidate records survive eligibility, decode, quality and overlap filtering, obtaining the central ordinary-record targets would require screening about **1.5–2.4M MiniCPM candidates** or **3.75–6M Spark candidates**. These yields are assumptions to replace with intake measurements. The 36-family catalog is ample for discovery; ingesting every family is not a requirement.

A storage illustration using 200kB/image, mono 24kHz/16-bit audio and 2Mbit/s compressed video gives approximately **334 GiB for MiniCPM / 836 GiB for Spark**, before episode assets, decoded frames, caches, checkpoints and backups. The existing 25 GiB proposed pilot ceiling is not a formal-corpus storage budget. Large media, weights and caches continue in ignored external/local stores; commit manifests and small evidence only.

## Charge the learning episodes explicitly

One reference episode contains 32 eligible teaching events, 8 future queries and 8 retention queries. It exercises the sixteen-event slow clock twice. Proposed accounting is:

```text
live observation passes                  = 32 × 256                =  8,192 positions
P-fast updates                          = 32 × 2                  =     64 steps
P-slow updates                          = (32 / 16) × 2            =      4 steps
inner passes including bounded replay   = (64 + 4) × 256           = 17,408 positions
future + retention outer query passes   = (8 + 8) × 512            =  8,192 positions
total per episode                                                   33,792 positions
```

This is a proposed accounting contract. It assumes an observation pass distinct from inner updates; if implementation reuses the same pass, remove that duplication. If it recomputes more context, add it. The four-observation differentiable `F` horizon remains separate from episode length. `P` uses the declared first-order approximation, not exact differentiation through 32 events of optimizer history.

Split teaching and future queries chronologically. Include new concepts/words, rules, corrections/revocation, unknown evidence, lexical-slot turnover, interference, audio patterns, AV relations and source-free transfer. Hold out underlying rule families, source recordings, actors and worlds, not only alias strings. An answer must remain expressible through base tokens after the lexical bank is cleared.

For the central recipe, total repeated inner `P` steps are **612,000 on MiniCPM / 1,530,000 on Spark**. These update selected plastic parameters. They are not that many full-backbone optimizer steps and are not new independent examples.

## Trainable parameters and optimization settings

“Training parameters” has two relevant meanings here:

| Quantity | MiniCPM | SparkX |
|---|---:|---:|
| Audited original backbone parameters | 2,516,756,480 | 4,112,079,360 |
| Complete-proposal planning envelope | about **2.8B** | about **4.4B** |
| Central outer position budget, excluding inner `P` passes | 1.817856B | 4.544640B |
| Outer updates at 32,768 packed positions/update | about **55,477** | about **138,692** |
| All-pass work divided by that batch, for cost comparison | 60,257.8 batch equivalents | 150,644.5 batch equivalents |

The envelopes are analytical placeholders, not measured final module inventories. The current MiniCPM augmented probe does not instantiate the entire proposal. All original tensors remain in the full joint training scope; active plastic experts and the full stored parameter count are different quantities.

Use pretrained encoders and backbones, zero-effect extension initialization where specified, packed/bucketed batches and the declared loss ledger. As a **development screen for the AdamW reference**, consider backbone rates around `5e-6–2e-5`, sensory encoders `1e-6–5e-6`, and new bridges/modules `1e-4–3e-4`, with short warmup and per-component monitoring. These are candidate ranges, not a selected recipe. Keep the current global clipping and numerical/state contracts. Tune inner learning rates separately on development episodes.

BF16 Muon is an outer-optimizer candidate under the [precision review](20261026_Mixed_Precision_Muon_and_Parallel_GPU_Budget_Review.md); its update scaling is different, so do not copy AdamW learning rates blindly. Inner `NativeFusedAdamW` remains the current FP32 reference. Neither BF16 nor the 3090 Ti compute unlock demonstrates fewer required training examples. Qualify optimization equivalence, retention and numerical stability before giving either a sample-efficiency credit.

## What would make the result convincing

Use at least three independent training seeds for finalists. Keep paired source groups across comparisons, independent natural recordings, held-out rule families, and both a controlled verifier suite and natural-media evaluation. Report uncertainty by source/world and seed; many questions, layers or repeated task orders from the same world do not create independent samples. Statistical tests must fit the metric and experimental design. [Dror et al., ACL 2018](https://aclanthology.org/P18-1128/).

The nested-learning plan already proposes **at least +10 percentage points over no-update** on new-instance acquisition, paired confidence intervals excluding zero, and **at most 2 points legacy-skill decline**. Keep the existing memory gates too: accepted correct rate ≥85%, accepted precision ≥90%, correction ≥60%, unaffected facts ≥70%, unknown false recall ≤10%, revoked false recall ≤25%, and oracle raw top-1 ≥85%. Meeting a gain target while failing these absolute gates is insufficient.

For planning, a paired binary comparison has approximate sample requirement

`n ≈ (z_(1−α/(2m)) + z_power)² × q / δ²`,

where `q` is the fraction of independent groups on which the two systems disagree, `δ` is the desired detectable gain, and `m` is the number of primary comparisons. With family α=0.05, power 80%, m=10 and q=0.2–0.4:

| Question | Approximate independent groups per comparison |
|---|---:|
| Detect a 10-point gain | 267–533; plan about **600** |
| Detect a 5-point gain | 1,066–2,131; plan about **2,200** |
| Establish 2-point retention noninferiority, q=0.05–0.15, true loss assumed zero | 1,460–4,380; plan about **4,500 per primary legacy skill** |

The 200-group development design detects roughly 11.5–16.3-point effects under these assumptions. It is a useful screen, not a strong test for a 5-point improvement. Use development discordance/cluster variance to refine the audit size; non-binary metrics, rare strata and near-margin retention need their own calculation. A true loss close to 2 points needs more samples than the zero-loss design. These calculations do not establish the probability that training will succeed.

Recommended sealed structure: about **2,200 independent groups per principal modality/learning endpoint**, with approximately **600-group alternate-order stress subsets** and a separate legacy-skill panel. Correct sensory substitutions must change answers appropriately. Show `P`-only transfer after clearing `E/F/A/C`, raw sources, KV, prototypes and learned lexical rows; compare `P` reset and wrong-world state. Demonstrate both ordinary-preparation versus episodic preparation and capacity/work-matched one-clock versus two-clock behavior. F6 additionally needs two successive policy improvements on fresh families at matched cumulative resources.

## Experiment count and compute consequence

Run one fixed-development scaling curve at **1/16, 1/4 and 1×** the proposed recipe, preserving mixture and full joint scope. The smaller points are diagnostics of the data curve, not a reduced final architecture. Promote only if held-out sensory dependence, new-instance learning and retention improve. If train loss falls while held-out results plateau, inspect leakage, labels and objectives before collecting four times more of the same data.

A useful *training subtotal* is three full seed runs, plus two alternative F4 preparation controls across three seeds at 20% of a schedule each, plus the 1/16 and 1/4 scaling runs: **4.5125 schedule equivalents**. This is **8.91B processed positions for MiniCPM / 22.28B for Spark**. It assumes controls can share each seed's identical pre-F4 checkpoint. Earlier-stage recurrent-core ablations or controls requiring different upstream preparation must be trained separately. It is not eight independent full retrains and it is not the total project budget.

The eight-arm `M/V/L` audit across three seeds, 2,200 canonical worlds and three alternate-order subsets of 600 worlds produces **96,000 episode visits**. Charging three extra query variants for state-reset controls gives **3.588B processed positions** with `P` inner updates only in four arms, or a **4.424B equal-work allowance**. Repeated worlds remain the same independent evaluation units. This is audit work, not training data. Its GPU-hours require measured small inner updates and forward-only queries; applying the bulk full-training rate directly would be misleading. Ordinary modality audits, legacy retention, extra preparation comparisons and policy evolution are additional.

For continuity, applying the earlier **conditional short-context** throughput assumptions to one central full-scope training schedule gives:

| Hardware | MiniCPM, 1.975B positions, GPU-h | Spark, 4.936B positions, GPU-h |
|---|---:|---:|
| DigitalOcean H200 | **179–652** | **679–2,471** |
| DigitalOcean H100 | 215–811 | 813–3,073 |
| DigitalOcean L40S | 655–2,150 | 2,439–8,048 |
| Two RTX 3090 Ti, stock | 1,319–4,270 | 4,885–15,872 |
| Two RTX 3090 Ti, qualified unlock | **837–3,140** | **3,094–11,594** |
| Two TITAN RTX, FP16 candidate | 1,363–4,459 | 8,938–32,303 |

GPU-hours are the sum across occupied devices: divide pair rows by two for wall-hours. These ranges inherit routine warmup/evaluation/checkpoint allowances; fixed qualification and the explicit large audit above are additional. A 4.5125-equivalent H200 training subtotal is approximately **810–2,943 GPU-h for MiniCPM / 3,066–11,149 for Spark**, before those additions. Reusing a dataset across seeds multiplies compute, not distinct data requirements.

These are conditional rescalings, not new CUDA measurements. They require the earlier memory/offload assumptions and a recheck of actual encoder load. TITAN has no native BF16 path; the prior report details FP16 and slower fallbacks. The four-GPU live pipeline requires connected gradients and can help capacity, while independent preparation jobs can overlap; neither automatically halves joint training work.

**The M2 Ultra still has no qualified executable budget for complete Spark training under the current guard.** Its short MiniCPM log cannot establish full-architecture BF16 throughput or fit. The previous M2 what-if range should not be turned into a promised finish date. This review changes the amount of work to plan, not that qualification status.

## Decision and replay

Retain the full architecture objective. First resolve the currently failed capability gates and qualify the one-clock text learner. Then run the smaller full-scope data curves, targeting the **2B MiniCPM / 5B Spark** central recipes only if the curves and component metrics support expansion. A 100M-position budget can produce useful mechanism evidence; 1B is an early integrated attempt; 5B is a reasonable funded Spark research attempt. None, by itself, establishes every product capability or autonomous recursive improvement.

Reproduce the arithmetic without model libraries, network access or media:

```sh
.venv/bin/python scripts/estimate_convincing_result_data_budget.py --output reports/local/data_budget_replay.json
```

The command verifies the 15 input hashes and refuses to overwrite output. The [research-source ledger](../../reports/convincing_result_data_budget_20261026/external_sources.json) records primary URLs and sections; [validation](../../reports/convincing_result_data_budget_20261026/validation.json) records arithmetic/replay checks. Source files are preserved in Git; no weight or dataset payload is added by this review.

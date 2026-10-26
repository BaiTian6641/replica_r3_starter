# MiniCPM5-2B: dataset intake and joint learning validation

**26 October 2026 · Prospective protocol design · No dataset admission or model execution.**

This companion turns the [dataset shortlist](20261026_SparkX_Dataset_Plan.md) into testable work packages for the [SparkX product architecture](../architecture/21_SparkX_2_6_4B_Consolidated_Architecture_and_MiniCPM_Validation.md). Existing execution protocols, resource guards and rejected capability dispositions continue to apply. A larger candidate data pool does not increase any current run's token/update budget.

## 1. Admission states and evidence

Keep these states explicit: discovered → metadata pinned → rights/source subset resolved → assets acquired → decoded and inspected → grouped/deduplicated → named splits admitted → trained → independently evaluated. The current review reaches **metadata pinned**, with selected primary-source checks. It does not reach asset QA.

Each admitted record needs the following fields, with source-specific additions where necessary:

| Field group | Required information |
|---|---|
| Identity | Dataset/release, HF commit, source record ID, immutable annotation digest, original media identity, local media-byte digest |
| Provenance | Original URL/recording, caption/QA parent, human/synthetic/derived annotation method, teacher/version when known |
| Rights | Annotation terms, per-asset terms, attribution, permitted-use disposition, reason for exclusion where unresolved |
| Split | Original split, source-group ID, train/dev/audit assignment, derivative cluster, duplicate checks and benchmark exclusion matches |
| Media | Duration, original sample rate/channels, decoded FPS/timebase, dimensions, crop/resize transform, resampling/normalization version |
| Supervision | Answer or label, point/track/time interval, annotator confidence/distribution, verifier type and version |
| Model-visible inputs | Raw sensory input, user prompt and legitimately supplied teaching feedback only; no answer-bearing file paths or verifier metadata |
| Cost | Download/decoded/cache bytes, audio seconds, video frames/crops, visual/audio/text positions, supervised positions, reuse count |

Keep targets, source scene graphs, masks, gold event IDs and label-coded filenames outside inference features. Geometry and source time transforms can be used to transform labels correctly; they cannot select a target crop from ground truth in the main perception arm. If using such a crop in an oracle diagnostic, label that arm explicitly.

Do not execute Hub loading scripts during source inspection. For later intake, prefer pinned Parquet/JSON and audited local decoding. Some inspected CREMA-D/RAVDESS mirrors rely on legacy Python loaders. The source files can be handled without granting arbitrary loader execution. Do not accept access agreements, initiate paid egress or substitute restricted media as a side effect of a generic download command.

## 2. First asset audit

For each selected source, inspect approximately **100 distinct upstream source groups**, stratified across task labels, rare categories, duration and annotation method. This is an engineering screening sample, not a powered estimate of all dataset quality. Report confidence intervals where meaningful; clustered questions must not masquerade as independent observations.

Proposed pilot admission rules:

- Every admitted asset has resolved provenance/rights, a valid decode and consistent coordinate/time metadata. Unknown or missing values are quarantined rather than silently filled.
- At least 95 of the 100 reviewed items are judged correct and usable for the declared task; report the interval and error categories. Failing a stratum triggers repair/exclusion and a fresh sample, not averaging it away with easier data.
- All tested identity/split joins are correct, and no known reserved benchmark group appears in the proposed training set. Exact duplicate/media-lineage violations are blocking.
- For reasoning subsets, at least 90 of 100 reviewed questions genuinely need the named modality or temporal evidence. This is a human screening criterion; subsequent model controls are still required.
- A small double-reviewed subset measures annotator disagreement. Ambiguous affect labels can be retained as distributions; invalid or unanswerable questions need explicit uncertain/abstain targets.

The 95/100 and 90/100 values are proposed screening thresholds, not published guarantees or product release gates. Adjust them prospectively by task after observing inter-rater reliability, before reviewing the final audit split.

Check actual decoded media. Recomputed audio duration outranks an unreliable annotation `duration` field. Some NVIDIA cards explicitly warn about this. A point must remain inside the transformed image, a temporal interval inside the correct source clip, and a track aligned to its intended object and frame sequence. Silently clipping invalid labels is not a quality repair.

## 3. Candidate pool and computational limits

The following is a **candidate pool**, larger than a single diagnostic run. Select a declared subset under the existing run caps. Do not assume that 10k long rationales fit a 2M-token pilot or that a small dataset proves full training convergence.

| Pool | Starting allocation | Why |
|---|---|---|
| Image grounding | 5k PixMo-Points + 3k reviewed PixMo-Cap examples | Concrete grounding plus rich perception, with shared images grouped |
| Video grounding/reasoning | 1k VideoPoint + 500 VideoTrack + 500 AskModelAnything examples | Temporal location, identity and open QA; each from resolved source subsets |
| Sound | Up to 5 hours from individually eligible FSD50K clips | Human event labels, balanced by class and source |
| Music fundamentals | Up to 2 hours NSynth training notes; optional source-verified Slakh controls replace part of this allocation | Pitch/timbre and temporal combinations with independent labels |
| Speech | Up to 2 hours from LibriSpeech clean training speakers/chapters | Keep speech alignment represented without dominating the acoustic curriculum |
| Perceived affect | Up to 1 hour of CREMA-D actor-grouped training data with original votes | Uncertainty and modality-specific perception |
| Audio reasoning | Up to 5 hours of rights-resolved, independently checked QA recordings | StrongAC/AudioMCQ source subset or newly authored QA over eligible audio; if no eligible subset exists, author reviewed QA rather than ingesting restricted material |
| AV controls | Up to 500 short owned or separately eligible AV episodes, outside the 15-hour audio pool | Pairing, synchronization, distractors and delayed event recall |
| Learning behavior | 1k fresh training episodes; initially 4–16 observations and 4–8 later queries per episode | Corrections, vocabulary turnover and transfer; multiple procedural/acoustic families required |

The visual allocation totals 10k examples; the audio ceilings total 15 hours. They are not independent-media counts until deduplication. A source asset can support several labels, but its repeated exposure must be counted. Adding OCR/chart/document examples should initially replace an equivalent measured budget, not silently enlarge it. Existing text-retention pools remain part of training.

Suggested acquisition partitions under the existing **25 GiB proposed asset ceiling** are 5 GiB images, 8 GiB video, 4 GiB audio and 8 GiB reserve for selected assets. This is a planning allocation, not an observed fit. Decoded intermediates/features and model checkpoints are separate; stop intake if the measured ledger exceeds its declared cap. An illustrative mono PCM estimate is `seconds × sample_rate × bytes_per_sample`: 15 hours at 24kHz/16-bit is about 2.41 GiB before containers, transcripts or alternative sample rates. Music may require higher rates; do not force one rate merely to match this example.

Minimize wasted computation by grouping QA siblings around one media encoding during eligible frozen-encoder diagnostics, bucketing duration, and bounding crops/frames. During full encoder fine-tuning, cached fixed features cannot replace live encoder forward/backward passes. A zero gate does not save compute unless execution is actually skipped. Preserve high-frequency/transient evidence when assessing whether audio pooling or sparse frame sampling is acceptable.

The architecture's provisional 30/20/20/15/15 mixture is by **processed model positions** for text, image/region, audio, temporal/AV, and memory/learning episodes, respectively, not row counts. For implementation, assign each sample a single primary mixture bucket, report all modality positions separately, and prevent an AV sample from being counted twice. Log supervised positions too: a long synthetic rationale can dominate loss even when its media share is small. Final bucket definitions and sampling weights belong in the prospective run manifest.

## 4. Validate multimodal competence before online learning

Full offline F2/F3 training must qualify the sensory pathways before F4 experience acquisition. F1 frozen-backbone warmups remain diagnostics. Use the same held-out source groups across paired controls; keep question text, prompt formatting and decoding budget constant.

| Capability | Positive task | Required control | Main metrics |
|---|---|---|---|
| Image grounding | Find/count/compare referenced objects, including absent cases | Blank image, matched wrong image, region removal, different crop | Point/count scores, grounded answer accuracy, false presence rate |
| Temporal video | Track identity; before/after/overlap; changed relations | Single frame, task-valid order reversal, shuffled frames, matched alternate event | Track/point metric, interval IoU or endpoint error, paired answer correctness |
| Sound understanding | Distinguish source/timbre/event with supporting evidence | Silence, matched wrong recording, same class with different answer-relevant pattern | Balanced accuracy, event scores, evidence consistency |
| Music reasoning | Instrument entry/order, pitch relation, pattern recurrence | Matched arrangement or instrument swap with recomputed labels | Pitch/order accuracy, event timing and held-out recording transfer |
| AV reasoning | Link audible event to visible actor/object across time | Muted audio, blank video, shifted sync, matched audio/video swaps | Joint versus unimodal accuracy, sync sensitivity, correct answer changes |
| Affect | Listener-perceived expression with uncertainty | Transcript-only, audio-only, video-only, conflicting cues | Distribution calibration, macro-F1, actor/corpus transfer; avoid a single hard-label score |
| Retention | Text, reasoning, instructions, code and prior perceptual tasks | Unmodified checkpoint under same decoding | Task-specific deltas and uncertainty; no aggregate score hiding failures |

Silence-induced score loss alone is insufficient: it can reflect a changed input distribution. Include **matched counterfactual pairs** in which the correct answer is known to change, and require the model to change correctly. Temporal reversal is useful only for tasks whose truth changes under reversal. Do not penalize a correct static answer for staying constant.

Compare concise evidence supervision with long rationale supervision at matched processed/supervised budgets. Synthetic explanations are training candidates, not direct observations of a teacher's internal reasoning or guaranteed correct acoustic descriptions. For latent visual reasoning, test whether the designated visual workspace/evidence route contributes beyond ordinary text decoding under matched compute; do not infer that mechanism from a fluent explanation.

Initial development sets should include at least 200 independent source groups per main modality where feasible. Report paired cluster-bootstrap intervals and raw counts; final product audit size needs a separate power/error-budget design. Evaluate held-out domains and real recordings alongside owned exact-verifier tasks. Results on one procedural renderer or a few controlled timbres are not enough.

## 5. Native experience episodes

Use a stable, already-qualified multimodal checkpoint. The episode generator may schedule teaching and hold verifier labels, but inference-time semantic selection, memory reads/writes, concept binding and learning decisions belong to the model. This preserves the native architecture objective.

One useful episode family is a novel acoustic pattern associated with a visible object and a taught name:

| Step | Model receives | Independent verifier retains |
|---|---|---|
| Observe | Several views and sounds, with distracting entities/events | Exact object/event identity, sound-source lineage and scene program |
| Teach | A novel name, a rule and contrasting examples | Correct concept mapping and rule extension |
| Correct | A correction to one relationship or a scoped exception | Old/new validity intervals and unaffected facts |
| Delay | Unrelated activity and controlled interference | Interference schedule and state budget |
| Transfer | New scene/recording/composition requiring the taught rule | Source-disjoint correct answer |
| Restart | Serialized permitted learned state | Expected continuity and proof of removed evidence |
| Forget/revise | Explicit revision or revocation | Targets whose answer must change, and targets that must remain stable |

For audio mixtures, compose only within a split; hold all variants of an instrument, original recording and MIDI composition together. For visual episodes, split world/renderer/object families before generating textual aliases. Include natural eligible media so procedural success has an external transfer check.

Proposed pilot counts are 1k training episodes, 200 development families and 200 sealed audit families. These are future generated/curated data, not available assets. Exact seeds, generator versions, held-out factors and verifier access must be frozen before experiments. A benchmark question cannot become a teaching example merely because it is wrapped in a dialogue.

## 6. Eight-arm joint test: memory, vocabulary and plastic learning

Use the existing architecture's factorial definition:

- `M`: intrinsic episodic/associative/addressing path (`E/F/A`) enabled versus gated off.
- `V`: swappable lexical bank enabled versus immutable base-token representation.
- `L`: persistent plastic-knowledge updates (`P`) enabled versus disabled.

Run all eight combinations with identical multimodal starting weights, episode order, feedback, evaluation questions and declared resource envelope. `L=0` means no **P updates**; an `M=1` arm can still update its associative state. Do not describe it as “no learning of any kind.” Always report which parameter/state types changed.

Primary comparisons are full `M+V+L` against its three single-component removals and against matched fixed-memory/base-vocabulary baselines. Examine interactions rather than adding three isolated gains. Report accuracy/retention, bytes, update time, replay work, active parameters, media re-encodes and total wall time. Both fixed-exposure and matched-total-compute comparisons are useful; predeclare the main one.

Required vocabulary tests include stable lexeme identity across slots, no collision after slot reuse, unchanged meaning after a lexical bank swap, new aliases for known concepts, new concepts with old words, and correct base-token fallback. A shorter output is not a semantic learning gain.

Use two different learning claims:

1. **Persistent memory:** retain the declared `E/F/A` state, remove original prompts/media/KV, restart, then test delayed use and corrections. This establishes the specified memory behavior if it passes.
2. **Persistent parameter transfer:** retain `P` plus stable model weights, reset `E/F/A/C`, clear old prompts/media/KV, explicit answer-bearing prototypes and learned lexical rows. Query new instances through base vocabulary. Compare against resetting `P` as well. This is the stronger test for acquired knowledge in the plastic component.

If only the first test succeeds, report memory improvement rather than parameter transfer. A mechanism may validly learn in `F`; test that separately with the retained-state boundary named. Neither result by itself proves recursive self-improvement. Policy improvement requires repeated fresh future-task families and better learning at matched total cost, as specified in the [intrinsic teaching plan](../research/sparkx_intrinsic_learning_spine_20261026/Teaching_and_Validation.md).

Evaluation examples may create temporary within-episode state when the benchmark explicitly requires it. They may not contribute persistent learning or feedback to later test examples/tenants. Snapshot/reset the relevant state at each independent evaluation boundary and discard benchmark-session updates after scoring.

## 7. Deliverables before a training run

Produce an exact asset manifest, exclusion/rights ledger, source-group split manifest, small media-QA report, normalization/version manifest, contamination report, target-loss ledger and measured storage/compute estimate. Then write the named prospective run protocol with scope, baseline, numeric gates, maximum updates/tokens and rollback conditions. Only admitted record IDs enter the runtime dataset registry.

Keep the final-product lineage distinct from restricted research experiments. Increasing data volume does not repair a collapsed classifier, wrong timestamp transform, invalid optimizer scope or absent modality dependence. Those failures require diagnosis before expansion. No new model-quality conclusion is claimed by this documentation update.

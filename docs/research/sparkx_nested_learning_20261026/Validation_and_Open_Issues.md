# MiniCPM first: validating the nested learner and closing design gaps

**Historical plan, superseded for active Spark text-memory work on 29 October 2026.**
Use [Spark memory S0–S1](../../experiments/20261029_Spark_Memory_S0_S1.md) and its
[dated evidence ledger](../../../reports/intrinsic_memory_v3/evidence_ledger_20261029.json).
The MiniCPM results and proposed NL ladder below remain historical/control evidence;
their multimodal prerequisites, sixteen-expert baseline and rejected native results
do not govern the current frozen-Spark, intrinsic-readout experiment.

**26 October 2026 · Proposed validation, not run authorization.** Use the [algorithm contract](Nested_Learning_Contract.md), [existing data curriculum](../../datasets/20261026_MiniCPM_Dataset_Validation.md), and [state/RSI contracts](../../architecture/22_State_Mechanism_and_RSI_Contracts.md) together. Existing stricter gates and execution limits retain precedence.

## 1. What is actually ready

The review captured repository HEAD `e05b0217808cec5f838672519bd368ec3beea0d5`, including separately hashed working-tree documents. Other work was active; this is an observation, not a claim that all referenced text was committed at that HEAD.

| Evidence | Current disposition | Consequence |
|---|---|---|
| Latest [vision balanced train04](../../experiments/20261026_Vision_Balanced_Alignment.md) | 24 projector-only updates; train-fit 3/14, dev 2/18, caption-following accuracy zero; stopped before sealed audit | Do not interpret successful gradients or optimizer steps as perception/reasoning competence. This diagnostic does not test the final full-FT proposal. |
| Earlier native memory `native04` | 110/240 recall, zero optimizer updates, learned quality rejected | Establish useful addressing/retention before attributing gains to higher-order learning. |
| Memory `restart04` | Fresh-process parity, zero re-encoding of old observations | Preserve this engineering property; it does not establish semantic transfer. |
| MiniCPM full-backbone probes | Short all-original-parameter update probes already exist | Demonstrates a narrow update path, not convergence or Spark resource fit. |
| [Dataset validation](../../../reports/sparkx_dataset_validation_20261026/validation.json) | Metadata-only pass; zero asset QA or training admissions | Actual media and episode admission remain prerequisites. |
| [First tranche](../../../reports/sparkx_first_tranche_20261026/tranche_receipt.json) | Six synthetic protocol fixtures, 1,798 bytes; no external media or training | Useful ingestion fixtures, not six trained or validated multimodal datasets. |
| [Spark residency v2](../../../reports/sparkx_residency_20261026/residency_v2.json) | Analytical accounting; loader/backend/resource gates unqualified | No native Spark full-FT or FP8 capability follows from the arithmetic. |

No new model, media acquisition or native training was run for this review.

## 2. Separate four claims

1. **Native access:** model computations read/write the intended state with correct timing and scope.
2. **Persistent learning:** experience changes a nominated parameter set and improves new-instance behavior after source removal and restart.
3. **Learning prepared by outer training:** post-adaptation performance exceeds an ordinary supervised-training control with matched data and compute.
4. **Learning-policy improvement:** a changed policy makes future acquisition more effective on fresh families, after task-specific state is reset.

A result may pass one and fail the next. More stored records, more confident answers, new parameter bytes, lower teaching loss or shorter prompts cannot substitute for these distinctions.

## 3. Sequential experiment ladder

### NL0 — deterministic learning and state contracts

Run inexpensive software/math checks before model training. Confirm per-expert moments and counters, selected-only updates, exactly-once observations, causal masks, transaction rollback, reset scope, vocab generations and cache invalidation. Check chunk-start reads and one shared write; a change in reader count must not multiply updates. Verify observer/spine absence parity.

For gradient ownership, use a tiny deterministic differentiable recurrence and finite differences. Demonstrate both direct outer gradients and the later-read-to-earlier-write path claimed for `F`. For `P`, compare the first-order estimator to a short exact mathematical AdamW recurrence and show the expected discrepancy; do not set a false gate requiring them to agree. Exact recurrence gradients must agree with finite differences. A fused update must match the declared reference values/moments on finite FP32 fixtures before it is used.

These are future implementation checks. This review's source/JSON/link/arithmetic verification is not NL0 model-runtime qualification.

### NL1 — text learning with a behavioral objective

Keep modality acquisition out of this mechanism test. Use fresh arbitrary associations **and** compositional rules with overlapping identities. Establish a sixteen-expert one-clock baseline against no-update, explicit-record recall and full-context controls. Original MiniCPM weights stay frozen during each teaching/test episode. Offline preparation is separate.

Teach, remove the teaching prompt and derived caches, restart, then test new examples. Retain only the named parameter state. For `P`-only evaluation retain `P` weights; remove `E/F/A/C`, source media, prototypes, active phrase rows and source KV. Reset optimizer moments for read-only evaluation if no more updates are allowed; for resumed-learning tests restore moments and counters exactly. Report those two tests separately.

Generate final nonce associations and world assignments after freezing the release, and keep their seeds outside candidate training. Hash original backbone/encoder tensors before and after teaching; the permitted changed set is explicit. In a `P`-only read test, the model must not read optimizer moments, teaching counters, replay metadata or expert-selection traces as an alternate knowledge channel.

Use pre-teaching `P`, wrong-world `P` and reset-`P` negatives. A correct novel spelling association is recall; transfer requires new instances or compositions. First prove any durable behavioral learning before expanding clocks or peripherals.

### NL2 — does outer preparation make learning better?

Compare ordinary supervised preparation and first-order episodic preparation from matched starting checkpoints, using the same teaching examples, future-query supervision budget and trainable scope. Count meta-query labels as training data. Include an equal-total-compute ordinary-training control; same token count alone does not match cost.

Measure post-adaptation gain, retention and learning curves against updates/seconds/teacher observations. Include a stronger zero-shot checkpoint control: a model that starts better may show a smaller absolute gain despite better final performance. Report both start and finish, not gain alone. A nested-learning claim needs better learning behavior on held-out rule families, not merely a better initialization on seen tasks.

The first-order gradient's missing paths remain explicit. Exact short-unroll model training is a later branch with separate protocol/backend qualification, not a hidden requirement that stalls all first-order investigation.

### NL3 — do two plastic clocks help?

Use the same sixteen experts, two active reads, teaching stream and replay bytes for:

| Arm | Purpose |
|---|---|
| One-clock bank | Capacity/active-work reference |
| Eight fast + eight slow, clock 16 | Primary proposed hierarchy |
| Same partition, one common clock | Isolate partition from timing |
| Swapped fast/slow assignments or clock phases | Detect schedule-specific shortcuts |
| Two-clock without replay | Isolate replay from schedule benefit |
| Fixed routing / inexpensive prototype routing | Detect gains attributable only to task identification |

Do not force every combination into a giant factorial. First eliminate configurations that fail acquisition/retention, then compare surviving candidates at matched total update work. A one-clock budget-matched arm distributes the same aggregate number of optimizer steps; also report its natural-rate result. Record actual active expert utilization and saturation.

Use at least 32 eligible teaching transactions per development episode so the slow boundary occurs twice. Evaluate before/after those boundaries and through 10/100/1000-event interference studies as feasible under admitted budgets. Report clock time as well as event count. A small event-count experiment cannot establish indefinite use.

### NL4 — can the cheaper local objective replace decoder backward?

Compare local feature learning with the behavioral reference on matched worlds and release scope. Count feature-extraction passes, cache construction, replay, optimizer work, acceptance checks and fresh-query evaluation. A regression loss on cached vectors is not a comparable unit to token-level cross-entropy.

Require the local route to pass the same source-free behavioral and retention gates and meet a preregistered noninferiority margin, initially at most 2 percentage points below the qualified behavioral reference on each required family. Its measured total learning cost must improve, with uncertainty reported. If it fails, do not label the expensive reference “computationally friendly” by ignoring backward work.

### NL5 — vision/audio experience after offline multimodal qualification

Full offline MiniCPM multimodal training precedes this stage. Require matched sensory controls, calibrated unknown behavior, spatial and temporal reasoning, and preserved text skills. Projector-only failure does not prove full FT will fail, but it cannot be waived by adding memory.

Then teach new visual categories/relations, recurring event rules, sound motifs, timbre/instrument or device patterns and audiovisual associations. Test new renderings, speakers/performers, recordings, views, backgrounds and compositions after source removal. Keep arbitrary facts and reusable rules as separate score families.

Repeat evidence interventions after each interference horizon: wrong image/region, shuffled frames, time reversal when relevant, wrong sound, silence, transcript-only, time-offset AV and changed acoustic attributes. Include harmless gain/view changes and missing-modality cases. Counterfactuals need valid matched targets; deleting all evidence may make the question unanswerable rather than change its answer.

Measure grounding retention alongside answer retention. A model that preserves answers by replacing sensory reasoning with a language prior fails the intended capability. The RCL-inspired audio extension is experimental; no published audio result is assumed. For perceived affect use held-out speakers and listener distributions, not a single purported true emotion. Functional appraisal cannot promote unverified content to knowledge.

### NL6 — memory, swappable vocabulary and plastic learning together

Retain the existing eight arms with precise factors:

- `M`: `E/F/A` pathway enabled versus disabled.
- `V`: swappable phrase rows versus base vocabulary.
- `L`: nominated `P` updates enabled versus identically initialized frozen `P`.

All arms retain the same installed architecture. `M=1,L=0` can still change `F`; it is not “no learning anywhere.” Add a true frozen-state control separately. Test the selected two-clock policy only after NL3, with its one-clock control carried into the finalist comparison.

Rotate the active vocab bank completely and deliberately reuse slots. Test meanings using base-token outputs, novel paraphrases and new visual/audio instances. Persist concept/lexeme IDs, not recycled slots. Keep candidate discovery independent of gold answers, and score decoded bytes so phrase packaging cannot inflate accuracy. Account for phrase selection/encoding and atomic-feedback costs.

Report main effects and interactions with uncertainty. A combination must improve the product-relevant outcome, not just each isolated module's private metric. Neither lower token count nor faster naming establishes learned meaning.

### NL7 — longevity, correction, efficiency and later policy improvement

Run repeated session restarts, mixed domains, overlapping tasks, source correction/revocation, failed updates and fixed-capacity interference. Compare multiple task orders: a canonical order, its reverse and at least two seeded permutations. Order effects are reported separately from training-seed effects.

Only after fixed-policy acquisition passes, introduce `Psi`: bounded choices of replay allocation, write eligibility or step allocation. Evaluate at least two successive candidate policy revisions on fresh held-out families, resetting task-specific knowledge for policy tests. Keep verifier criteria and protected identity conditioning outside candidate write authority. The objective is better **future learning per total resource**, not merely better current-task answers.

## 4. Dataset/episode schema additions

The 36-family catalog supplies candidate sources, not nested-learning episodes by itself. Extend every episode manifest with:

| Field group | Required content |
|---|---|
| Identity | `world_uid`, `scope_uid`, session IDs, concept/rule family, source-recording/performer/asset group |
| Split | outer-training/development/sealed-final role; teaching versus future-query role; contamination group |
| Time | observation time, event interval, actual availability time, causal cutoff, eligibility counter and clock phase |
| Supervision | feedback type, independent verification provenance, uncertainty/listener perspective, target availability time |
| Learning | allowed mutable tensors, optimizer rule and state policy, step/vector/position limits, replay sample IDs |
| Semantics | stable concept and lexeme IDs, base-token spelling, phrase slot/generation only as transient presentation |
| Interventions | source-removal list, retained/reset state classes, wrong-world control, modality substitutions and their targets |
| Restart | release/representation/state hashes, moments/counters, pending consolidation and deduplication keys |
| Accounting | raw/decoded bytes, frames/seconds, positions/vectors, elapsed subphases, peak bytes and teacher requests |

Split at world/rule/recording/performer level before deriving captions, QA, augmentations or replay. Keep future-query labels out of teaching and candidate selection. Final held-out families cannot tune loss weights, clocks, learning rates, vocab candidates or acceptance thresholds. Real-media asset audits and rights/provenance checks remain required before admission.

## 5. Proposed promotion criteria

Use at least three independent seeds for the finalist comparison, paired worlds across arms, and confidence intervals clustered by world/recording. A power calculation from development variance chooses final sample counts; repeated questions from one clip are not independent samples. The primary endpoint and exclusions are registered before sealed evaluation.

- **Persistent acquisition:** at least 10 percentage points over no-update on the designated new-instance endpoint, with paired 95% confidence interval excluding zero; report absolute performance and floor/ceiling caveats.
- **Retention:** at most 2 percentage points decline on qualified legacy skills, plus every existing stricter capability gate. An aggregate cannot hide failed correction, unknown handling or sensory dependence.
- **Nested preparation:** better post-adaptation behavior than matched ordinary preparation, without an unreported cost or data advantage. Failure means the mechanism may still be useful but the stronger learned-learning claim is unqualified.
- **Two clocks:** a quality/cost Pareto benefit over capacity- and work-matched one-clock controls; no mandatory claim that two clocks must win.
- **Causality/persistence:** no future-target leakage, no duplicate writes, exact scope isolation, correct resumed optimizer trajectory and qualified fresh-process restoration.
- **Efficiency:** retain proposed warm text P95 overhead ≤10%, online-learning duty cycle ≤5%, and AV streaming real-time factor ≤1 on a declared workload. Report P99 stalls, queue growth and maintenance bursts too. These are targets, not measurements.
- **Policy improvement:** repeated gains on fresh learning families with fixed-policy and equal-budget controls, including failed candidate costs.

For a soak period `T`, define learning duty as total serialized learning/maintenance time divided by `T`, with overlapping intervals counted once. Separately report device-active time and actual serving latency under concurrency. Work moved off the main thread is not free. Initial full-FT cost, all future learning, replay and release migration enter the break-even comparison with repeated full fine-tuning.

## 6. Issue ledger: closed design decisions and open empirical obligations

“Specified” means a concrete decision/test exists. It does not mean implementation or evidence is complete. Owners below are work areas, not invented assigned people.

| ID | Issue and decision | Evidence required / owner | Status |
|---|---|---|---|
| N01 | Multiple modules were being conflated with nested learning; use four claim levels | NL1–NL3 behavior / learning research | Specified; unproven |
| N02 | Missing plastic rates; use 8+8 experts and clock 16 against one-clock controls | NL3 result / architecture | Specified; unrun |
| N03 | Optimizer-policy conflict; fused AdamW replaces prospective SGD recipe | Selected-only/resume/parity receipts / runtime | Specified; implementation pending |
| N04 | Fused optimizer lacks exact meta-gradients; first-order path is explicit | Gradient audit and future-query gain; exact path separately qualified / training | Specified limitation |
| N05 | Local loss might not teach useful behavior | NL4 noninferiority and measured cost / learning research | Open empirical |
| N06 | Slower clock absent from short episodes | 32-event episodes and logged slow-boundary evaluations / data | Specified; data pending |
| N07 | Shared ports could duplicate writes or lose credit | NL0 causality and historical-gradient tests / runtime | Specified; implementation pending |
| N08 | Memory survival confused with transferable knowledge | Named storage interventions and new instances / evaluation | Specified; unrun |
| N09 | Feature/representation drift could corrupt replay | Pinned producers; migration invalidation tests / data/runtime | Specified; unrun |
| N10 | Slot recycling could corrupt concept identity | NL6 turnover and base-token semantic tests / vocab | Specified; unrun |
| N11 | Full multimodal skill not established locally | Sensory dependence and full-FT quality gates / multimodal training | Open prerequisite; current quality rejected |
| N12 | Correct answers could hide grounding loss | NL5 counterfactual retention / evaluation | Specified; unrun |
| N13 | Task separation/order could inflate transfer | Overlapping concepts, cheap router, four orders / data/evaluation | Specified; unrun |
| N14 | Surprising/affective content could drive false learning | Noise, wrong feedback and appraisal ablations / core | Specified; unrun |
| N15 | Personality content unavailable | User-supplied description; stable identity/null-path tests / identity | Pending input; other work can proceed |
| N16 | 128 MiB profile omitted resident moments | New 256+64 MiB candidate; actual device peak / runtime | Arithmetic corrected; fit unmeasured |
| N17 | Spark full-FT fit not demonstrated | Native loader/backend and sustained resource qualification / provisioning | Open prerequisite |
| N18 | Metadata/fixtures could be mistaken for real dataset readiness | Asset QA, episode sealing, training admission / data | Open prerequisite |
| N19 | Compressed knowledge cannot guarantee exact source erasure | Restore/replay fallback, deletion limits and correction tests / persistence | Limitation explicit; protocol unrun |
| N20 | Fixed capacity may saturate | Age/novelty curves, eviction and bounded growth / architecture | Open empirical |
| N21 | UAP could absorb intrinsic memory responsibilities | Empty-spine and observer-parity tests / integration | Boundary specified; unrun |
| N22 | Continual learning could be mislabeled recursive improvement | NL7 fresh-family policy tests and document 22 RSI gates / learning policy | Later research |

The next implementation should address NL0 and the current single-capability failures, then a small NL1/NL2 text study. Full audiovisual self-learning and recursive policy change are later stages. Nothing in this review authorizes skipping existing model, data, execution or resource gates.

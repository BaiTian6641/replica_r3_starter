# 04 — Datasets, authentic source records, and generation instructions

**TAIS / Replica R3 · 23 October 2026**  
**Evidence status:** official release cards and author repositories reviewed; no public dataset has been ingested or model-trained in this bundle. Counts in the pilot plan are proposed budgets. Dataset licenses and revisions must be retained when a run actually fetches them.

**24 October expansion:** use [the acquisition/curriculum plan](08_Dataset_Acquisition_and_Curriculum.md)
and [34-source catalog](datasets/20261024_Source_Catalog.json) for the current
visual/text/memory intake queue. Twenty-two sources now have immutable metadata/
card pins; no dataset assets were downloaded or newly admitted. That plan's
staged quantities and source-specific gates supersede this guide's older pilot
budgets where they differ. The authentic-source and lineage rules below remain.

## 1. What data this project needs

The project needs three different kinds of evidence. Public training data can teach general reasoning, memory use and action selection. Independently generated environments can reveal whether experience produces a transferable improvement. The source person's own reviewed decisions and recollections initialize the Replica. These sources answer different questions, so keep their identity and lineage visible in every experiment.

A useful memory example teaches how to resolve a changed fact, ask when evidence is missing, preserve uncertainty, or apply a remembered procedure to a new case. A useful Replica example teaches what information the person considers decisive and what would reverse their decision. Neither objective is served by flooding the model with invented autobiographical stories.

**Initial recommendation:** use PersonaMem-v2 training personas for generic personalization; a small source-filtered Tülu/OpenThoughts sample for competence replay; a small TOUCAN slice for tool behavior; and independently generated, solver-checked episodes for temporal memory and procedural learning. Keep LongMemEval cleaned/V2, LoCoMo, BFCL and designated PersonaMem benchmark splits outside training. This is an engineering selection, not a published optimal mixture.

## 2. Verified source catalog

License entries identify the inspected release. Preserve component licenses and third-party output terms; a collection's top-level label does not erase upstream conditions. The starter contains links and metadata guidance, not redistributed dataset contents.

| Dataset and official source | What the release contains | Use in this project | License, version and practical limits |
|---|---|---|---|
| **[PersonaMem-v2](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2)** · [code](https://github.com/bowen-upenn/PersonaMem-v2) · [paper](https://arxiv.org/abs/2512.06688) | Implicit and evolving preferences; long text and multimodal histories. Config `benchmark`; text train 18,500 / val 2,600; multimodal train 19,200 / val 2,110; separate 5,000-query benchmarks. | Priority generic personalization training and validation. Use the official training identities and preserve their separation. | Card: CC-BY-4.0. Synthetic people are not this user's autobiography. Audit cross-modal overlap by persona ID. Read actual image assets for visual experiments. |
| **[PersonaMem-v3](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v3)** · [code](https://github.com/bowen-upenn/PersonaMem-v3) · [paper](https://arxiv.org/abs/2608.21381) | Cross-platform histories and tasks, including preference lifecycle, proactivity, restraint and forgetting directives; GIST-Bench-derived behavioral grounding. | Additional research evaluation after the first local loop works. No designated training split was identified in the inspected release. | Card: CC-BY-NC-4.0 inherited from GIST-Bench. The authors require masking engagement events at or after the query time. Treat the reconstructed personas according to their stated provenance. |
| **[LongMemEval cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)** · [code](https://github.com/xiaowu0162/LongMemEval) · [paper](https://arxiv.org/abs/2410.10813) | 500 question identities across S/M/oracle settings: extraction, multi-session reasoning, updates, time and abstention. | External memory evaluation; oracle retrieval isolates reader failures. | MIT card/repo. The old `xiaowu0162/longmemeval` is deprecated. Use cleaned S/M files. Variants reuse question identities and cannot be split as independent samples. |
| **[LongMemEval-V2](https://huggingface.co/datasets/xiaowu0162/longmemeval-v2)** · [code](https://github.com/xiaowu0162/LongMemEval-V2) · [paper](https://arxiv.org/abs/2605.12493) | 451 manually curated questions and 1,870 task trajectories from customized web/enterprise environments; text observations and screenshots. Tests workflow knowledge, state changes, environment-specific failures and invalid premises. | Strong later evaluation of experience-derived procedures and multimodal memory. Begin with the small tier and measure query latency as well as correctness. | Apache-2.0 data/card/repo. Use official `questions.jsonl`, `trajectories.jsonl` and haystack mapping files. The HF viewer's auto-detected `train` image split is not an authorized training split for this evaluation benchmark. |
| **[LoCoMo](https://github.com/snap-research/locomo)** · [paper](https://arxiv.org/abs/2402.17753) | `data/locomo10.json`: ten long conversations, question answers, event summaries and multimodal-dialog annotations. | Secondary long-horizon memory evaluation. Cluster uncertainty estimates by conversation. | [CC-BY-NC-4.0](https://github.com/snap-research/locomo/blob/main/LICENSE.txt). Ten conversations provide few independent groups. Image URLs and generated captions require availability checks; captions alone cannot validate visual memory. |
| **[LaMP](https://lamp-benchmark.github.io/)** · [code](https://github.com/LaMP-Benchmark/LaMP) · [paper](https://arxiv.org/abs/2304.11406) | Seven personalization tasks with task-specific train/dev/test releases. | Optional personalized ranking and generation supplement. Choose a task and record its precise split. | Repository code/data-creation methods: CC-BY-NC-SA-4.0; constituent datasets retain their own licenses. Task performance is a personalization proxy, not a measure of whole-person fidelity. |
| **[Tülu 3 SFT mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)** · [code](https://github.com/allenai/open-instruct) | `train`: 939,343 rows in the 24 October pinned metadata (the earlier card count was 939,344), with `id`, `messages` and `source`. | Small source-balanced general competence replay. | Collection ODC-BY-1.0; subset-specific terms include noncommercial portions. Retain `source` and use an approved subset allowlist. The card also identifies third-party-generated outputs. |
| **[OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M)** · [code](https://github.com/open-thoughts/open-thoughts) · [paper](https://arxiv.org/abs/2506.04178) | `train`: math/code/science annotations; its recipe samples 75,000 questions with 16 QwQ-32B completions each. | A short, verified reasoning slice for replay and procedural transfer. | Card: Apache-2.0; retain source/provider terms. All completions of the same underlying question stay in the same split. Select complete short traces or regenerate verified concise solutions; never cut a reasoning trace at an arbitrary token limit. |
| **[UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback)** · [code](https://github.com/OpenBMB/UltraFeedback) · [paper](https://arxiv.org/abs/2310.01377) | `train`, approximately 64,000 prompts, candidate responses and model feedback. | Optional preference-ranking warm start after supervised training is measurable. | MIT card. Model ratings are preference signals, not ground-truth correctness or calibrated self-knowledge. Preserve upstream prompt and output terms. |
| **[Cleaned UltraFeedback pairs](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned)** | `train`, about 60,900 chosen/rejected pairs. | Optional DPO/control arm with inspected pair filtering. | MIT card. This is a transformation of UltraFeedback, not an independent corpus. Keep shared prompt identities together. |
| **[TOUCAN](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M)** · [code](https://github.com/TheAgentArk/Toucan) · [paper](https://arxiv.org/abs/2510.01179) | MCP tool trajectories; different teacher configurations and a curated SFT subset; tool schemas, single/multi-turn and irrelevant-tool cases. | Small, filtered tool-selection and recovery slice. Replay against local fixtures. | Apache-2.0. The card retains some partially failed trajectories; quality scores are model judgments. PII removal is best effort. Tool responses reflect the collection period, so they are not current facts. |
| **[Berkeley Function Calling Leaderboard](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard)** · [code](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) | Per-category evaluation JSONL and official evaluators. | External function-selection and multi-turn evaluation. | Apache-2.0 card. Pin exact BFCL version; the inspected HF card documents V1–V3 and directs users to its loader rather than generic `load_dataset()`. Check the current official repo before a later-version run. |
| **[CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/)** · [author generator](https://github.com/facebookresearch/clevr-dataset-gen) | Synthetic images, scene graphs, questions and functional programs. | Controlled visual reasoning; independently generated chronological scene changes are a useful proposed extension. | Dataset site: CC-BY-4.0. The generator has separate license/patent files and old Blender dependencies. A temporal extension is our experiment, not an existing CLEVR memory result. |

### Exact initial acquisition targets

For PersonaMem-v2, the [official README metadata](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2/blob/main/README.md) names configuration `benchmark` and these splits:

| Split | File |
|---|---|
| `train_text` | `benchmark/text/train.csv` |
| `val_text` | `benchmark/text/val.csv` |
| `benchmark_text` | `benchmark/text/benchmark.csv` |
| `train_multimodal` | `benchmark/multimodal/train.csv` |
| `val_multimodal` | `benchmark/multimodal/val.csv` |
| `benchmark_multimodal` | `benchmark/multimodal/benchmark.csv` |

The CSVs reference the histories; loading question rows alone does not load the episodic context. Inspect the revision's file layout and causal cutoff fields before writing its adapter.

The [LongMemEval author repository](https://github.com/xiaowu0162/LongMemEval) currently directs to `longmemeval_s_cleaned.json`, `longmemeval_m_cleaned.json`, and `longmemeval_oracle.json` in the cleaned dataset repository. Its label fields include evidence locations. Keep those fields on the evaluator side.

Do not batch-download every catalog entry. Fetch the selected revision, retain its license/card and hashes, inspect a few complete records, then write one adapter. Record unsupported or ambiguous formats as unresolved; a list of links is not an ingestion pipeline.

## 3. A manageable proposed pilot

Run a 2,000-target infrastructure slice first. It tests formatting, masking, memory causality, gradients and checkpoint restart. After it passes, use a **20,000-target text pilot** with a planned **8–32 million processed training-token budget**. This is a cap for the selected training windows and memory inputs, not a claim that every original long history fits in that budget. Full history ingestion, teacher generation and evaluation costs are measured separately.

| Component | Proposed targets | Preparation |
|---|---:|---|
| General competence replay | 8,000 | Source-allowlisted Tülu examples, balanced by task |
| Verified reasoning | 3,000 | Short complete examples; code/math verifier where applicable |
| Temporal memory and correction | 4,000 | Independently generated worlds with exact state transitions |
| Generic personalization | 2,000 | PersonaMem-v2 training identities only |
| Tool use and recovery | 1,000 | Filtered TOUCAN or local tool fixtures |
| Ask/retrieve/verify/stop control | 2,000 | Tasks with explicit information gaps and action costs |
| **Total** | **20,000** | Hold counts fixed for initial architecture comparisons |

Set split assignments at the **source group** level before generating any paraphrase. Public official validation/test material never contributes training targets. Create independent project groups for development, calibration and promotion; maintain a final audit pool unseen by teachers and the learning director.

For the VLM stage, add a separate proposed 3,000-target tranche: 2,000 PersonaMem-v2 multimodal training cases and 1,000 independently generated visual chronology cases. Use actual image tensors. Compare image-enabled, text-only, image-shuffled and image-removed conditions. This additional tranche has its own token/image budget and does not silently inflate the text arm's training compute.

For the Replica, collect 30–100 authentic source decision episodes gradually. Hold out complete topic families and later decisions. Teacher paraphrases are descendants of those cases, not new independent human observations. Use a low mixture share or a separable personal adapter initially; select the share from measured fidelity and general competence rather than repeatedly cycling a tiny personal archive until its training loss is near zero.

## 4. Pipeline from source to training

### 4.1 Freeze provenance and roles first

Each source registry entry should contain a canonical source ID, URL, exact revision, acquisition date, file hash, license reference/hash, allowed uses, source type and dataset role. Each derived item keeps its parent source IDs, the teacher/version/prompt identity and its group assignment. Store these rich records outside the minimal trainer export.

Use these roles consistently:

- `train`: eligible learning examples.
- `dev`: debugging and model/configuration selection.
- `calibration`: fitting a confidence or action threshold.
- `promotion`: independent comparison of a candidate against the current version, with a predefined exposure policy.
- `audit`: final sealed reporting; never passed to a teacher or a learning director.

No amount of renaming makes a used audit test sealed again. If a result informed subsequent design, document the exposure and use a fresh independent audit group for final claims. Repeated promotion also creates selection bias; bound the number of candidate comparisons or replenish independent groups under evaluator control.

### 4.2 Group before splitting

Group by real source decision, source document, synthetic world and rule family, persona identity, original reasoning question, or tool-task seed. All paraphrases, teacher variants, translations, multimodal versions and preference pairs inherit the same group and split. For a public dataset with official splits, those split restrictions take priority.

If text and visual releases use the same persona across conflicting roles, quarantine the conflicting group or reserve the whole group for the more restricted evaluation role. Do not resolve the conflict by moving official test targets into training.

### 4.3 Separate available evidence from gold labels

A memory input must be something the model could have observed before the answer is due. Track event time and observation time separately. A late report about an earlier event becomes available when the report arrives. It cannot appear in an earlier prediction's memory.

For public datasets, annotation fields such as `correct_answer`, preferred choice, `has_answer`, evidence-session IDs, raw oracle profiles and answer-revealing summaries are evaluator labels. Some source datasets provide both ordinary history and privileged annotations; use only the former as normal inputs. A naturally occurring statement in history is valid evidence even when it later answers a question. A gold-answer field copied into memory is leakage.

### 4.4 Generate the world, then verbalize it

For initial procedural tasks, generate a small deterministic world before asking an LLM to write dialogue. Examples include units and conversions, a fictional hardware-register map, constrained scheduling, a small packet format, a finite-state machine, or a visual scene graph. A solver computes outcomes; the teacher makes the task varied and readable. Avoid live hardware dependencies or private real-world systems in this first experiment.

The teacher can propose a useful practice target after a failure. The verifier decides whether its examples are correct. A separate evaluator decides whether training improved future behavior on unseen cases. Those are three distinct decisions.

### 4.5 Review and promote examples

Run deterministic checks on schema, duplicate IDs, group/split consistency, dates, source references, consent and executable answers. Review a stratified sample of accepted and rejected records, including correction, uncertainty, contradiction and failed-tool cases. As an initial operational choice, inspect at least 20 cases from each critical category and report observed error rates; this sample is not proof that all rows are correct.

A second LLM critic can find problems but does not make labels independent ground truth. For synthetic worlds, use a solver. For real evidence, check cited spans. For source-person interpretations and predicted counterfactual choices, request source review when not already explicit. Reject unsupported records rather than repairing them by invention.

## 5. Exact starter trainer export

The supplied trainer-facing data contract is one JSON object per line with exactly these fields:

```json
{
  "id": "world17.query08",
  "group_id": "world17",
  "split": "train",
  "prompt": "Which connector was the test fixture using as of Tuesday?",
  "response": "Connector B. The Tuesday maintenance record supersedes Monday's connector A record.",
  "memory_texts": [
    "event17a | Monday | Synthetic fixture used connector A.",
    "event17b | Tuesday | Technician replaced connector A with connector B."
  ],
  "source_ids": ["event17a", "event17b"],
  "verified": false
}
```

Allowed splits are `train`, `dev`, `calibration`, `promotion`, `audit`. The generator always emits `verified: false`. A separate checker produces validation evidence; the preparation process can then set this field to true for accepted records. It must not equate a teacher's confidence, a second model's agreement, or syntactic JSON validity with verification.

`memory_texts` contains permitted, causally available evidence. `source_ids` points to registry records; it does not authorize a lookup that bypasses a split or time boundary. `prompt` and `response` are strings in the final export. The richer canonical episode may contain events, timestamps, modalities, consent, verifier results and provenance, but it is converted explicitly to this export rather than silently stuffing evaluator metadata into model inputs.

**Current text-export boundary:** these eight fields do not include image paths or tensors. The multimodal generation prompt below produces candidates that refer to asset IDs and an external manifest; these are planning and text-control records. Actual VLM training needs the separate model-specific multimodal collator and causal asset loader described in the architecture plan. An image reference written into `memory_texts` is not evidence that pixels reached the model.

Keep private source records and their manifest local. Their presence in an export is not permission to send them to an external teacher. Use the input packet's existing consent scope; missing external-use permission means run locally or omit the record, without inferring permission from the dataset task itself.

## 6. Prompt pack

Each file is a complete instruction block with an input placeholder, small batch limit and bounded output. Fill its input packet with permitted records and stable split assignments. Do not provide audit data. The prompts create candidates; they cannot authorize their own admission to training.

| Prompt | Purpose |
|---|---|
| [01_Source_Grounded_Expansion.md](../prompts/01_Source_Grounded_Expansion.md) | Expand supported episodes without adding facts |
| [02_Temporal_Correction_and_Forgetting.md](../prompts/02_Temporal_Correction_and_Forgetting.md) | Updates, historical queries, expiration and deletion behavior |
| [03_Ask_Answer_and_Verify.md](../prompts/03_Ask_Answer_and_Verify.md) | Ask vs answer vs retrieval and checking |
| [04_Verified_Microengineering.md](../prompts/04_Verified_Microengineering.md) | Small, executable engineering tasks with independent expected outputs |
| [05_Failure_Analysis_and_Replay.md](../prompts/05_Failure_Analysis_and_Replay.md) | Learn from verified errors and preserve counterexamples |
| [06_Replica_Decisions_and_Counterfactuals.md](../prompts/06_Replica_Decisions_and_Counterfactuals.md) | Authentic decision extraction and clearly marked hypothetical variants |
| [07_Multimodal_Chronology.md](../prompts/07_Multimodal_Chronology.md) | Image-grounded observation histories and ablations |
| [08_Independent_Review_and_Split_Audit.md](../prompts/08_Independent_Review_and_Split_Audit.md) | Independent evidence review, leakage and split audit |

Use a teacher's concise evidence explanation when it improves supervision. Do not treat a generated explanation as a literal transcript of human cognition. Record the model/version and prompt used so that the dataset can be rebuilt and biased teacher patterns can be investigated.

## 7. Acceptance rules for the first real dataset

Before admitting an example, the ingestion process must confirm all of the following:

1. A source registry entry exists and permits the intended use.
2. Source and derived items have consistent group/split assignments.
3. All memory evidence is available by the query cutoff; future updates remain hidden.
4. Source-person facts, agent experience, inference and synthetic scenarios retain their ownership.
5. The target is supported by evidence, a solver result, or an explicit unknown/ask condition.
6. Gold annotations and answer-revealing metadata do not appear in model inputs.
7. The output fits the target tokenizer/template without truncating a necessary solution or corrupting tool syntax.
8. Independent verification evidence exists before `verified` becomes true.

For genuine source memory, preserve uncertainties and corrections. A deletion should immediately remove retrieval eligibility and invalidate derived artifacts according to their lineage. Removing a record from a dataset does not establish removal of its influence from a trained checkpoint. Rebuild from a permitted checkpoint when that guarantee is required; evaluate any proposed unlearning method separately.

## 8. What will make the resulting data convincing

Report accuracy and failure types by task family, memory age, correction count and evidence availability. Evaluate delayed queries after restart. Measure whether a procedure transfers to new values and new compositions. Test counterfactual decisions where one decisive constraint changes and paraphrases where no decisive constraint changes. Include deliberate ambiguity and conflicting evidence so that correctness does not depend on always producing an answer.

For self-improvement, compare a fixed training schedule with a learning director's proposed curriculum under the same data and compute budget. Keep the external evaluator fixed. Better training loss, more stored memories or more fluent explanations alone do not establish improvement. The key evidence is a verified gain on later unseen tasks, with retained general competence and source fidelity.

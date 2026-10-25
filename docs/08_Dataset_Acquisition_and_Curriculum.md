# Dataset acquisition, evidence and curriculum

**Prepared 25 October 2026 for the `7c7018e` design review.** This plan expands
the [existing data guide](04_Datasets_and_Generation.md) and supports the
[integrated design](architecture/19_Integrated_Memory_Learning_and_RSI_Design.md).
It catalogues **34 external sources: 22 with immutable Hub revisions and reviewed
cards, plus 12 primary-source candidates awaiting immutable pins**. Two additional
internal corpora are proposed. This update collected metadata/configuration and
cards, not dataset archives, training examples or model weights.

The [machine-readable catalog](datasets/20261024_Source_Catalog.json) records
revision/card hashes, source links, proposed role, grouping and unresolved terms.
All new entries have `auto_admit=false` and `training_admitted=false`. The
existing [runtime registry](../configs/dataset_registry.json) remains the
experiment-admission interface; this research catalog does not override it.

## 1. Intake states and ownership

Track each source through **located → metadata pinned → assets acquired → content
inspected → normalized/deduplicated → admitted for a named split/run → trained or
evaluated**. Never collapse these into a single “dataset ready” flag. Pinning a
card establishes source identity, not image rights, label quality or access to
every underlying component.

Each intake manifest records upstream repository/revision, original asset URL
or ID, selected config/split/shard, component terms, file digests, normalization
version, sample seed/order, rejected rows and resulting row/asset/family counts.
Record package attribution and any research-only/noncommercial restrictions.
Unknown terms remain pending; availability through a mirror is not independent
permission. Substitute eligible sources or owned procedural renders where needed.

The current small memory fixtures remain exposed regression data. Acquire or
generate fresh independent families before new capability claims. External
benchmarks remain reserved unless their official train split is explicitly
admitted; test/dev labels are never silently folded into practice generation.

## 2. Pinned memory, text and learning sources

Full 40-character revisions and card hashes are in the catalog. Sizes below are
upstream descriptions or metadata, not local asset inventories.

| Source | Intended role | Important admission/split rule |
|---|---|---|
| [PersonaMem-v2](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2) | Persona/temporal method training from declared train splits; reserved benchmark | CC-BY-4.0 card; group complete personas and all histories. Public persona data are not authentic user identity facts |
| [PersonaMem-v3](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v3) | Evaluation only | CC-BY-NC-4.0; preserve whole profiles, query families and multimodal provenance |
| [LongMemEval cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) | Long-term memory evaluation | MIT card; reserve complete histories/questions |
| [LongMemEval v2](https://huggingface.co/datasets/xiaowu0162/longmemeval-v2) | Separate versioned memory evaluation | Apache-2.0 card; decontaminate overlapping users/histories against the cleaned release |
| [Tulu 3 SFT mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) | Text instruction/retention replay | ODC-BY collection plus component terms; pinned metadata has 939,343 rows, distinct from the older guide's quoted count |
| [OpenThoughts3-1.2M](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) | Independently checked reasoning practice | Apache-2.0 card; roughly 1.2M completions from 75k questions, so all sibling solutions share a split |
| [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) | Optional later preference study | MIT card; teacher scores are preferences, not fact verification |
| [Toucan-1.5M](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M) | Tool-use practice candidate | Apache-2.0 card; validate executable tool schemas and replay outcomes |
| [Berkeley Function Calling Leaderboard](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) | Tool evaluation only | Apache-2.0 card; pin version, evaluator and schemas; reference calls stay outside training |
| [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) | Capped instruction/code/constraint replay | Component-specific terms; allowlist useful verified subsets rather than ingesting the whole mixture |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | Deferred language/CPT reserve | ODC-BY collection with underlying source conditions; outside the initial V0/V1 work |

## 3. Pinned visual sources

| Source | Intended role and upstream scale | Important admission/split rule |
|---|---|---|
| [LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) | Candidate V1 alignment pool, 558k pairs | Card license `other`, inherited LAION/CC/SBU terms and research-purpose image-bundle restriction; not automatically admitted |
| [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) | Candidate V2 instructions | CC-BY-4.0 card plus COCO image and generation/provider terms; group questions by original image |
| [The Cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) | Visual instruction collection, 50 component datasets | No blanket license; component allowlist and cross-source dedup required |
| [Docmatix](https://huggingface.co/datasets/HuggingFaceM4/Docmatix) | Document/visual reasoning subsets | MIT card plus source-asset terms; choose `images`, not both `images` and `pdf`. Pinned image config has 1,273,215 rows; a row may contain several images |
| [FineVision](https://huggingface.co/datasets/HuggingFaceM4/FineVision) | Broad later visual pool; card reports 24.3M samples | Per-component terms; multi-terabyte collection, subset only; overlaps individual datasets and other mixtures |
| [VQAv2 mirror](https://huggingface.co/datasets/HuggingFaceM4/VQAv2), [original](https://visualqa.org/download.html) | Object and scene grounding | Resolve original annotation and COCO asset terms; official train split only for training, image-level decontamination |
| [ChartQA mirror](https://huggingface.co/datasets/HuggingFaceM4/ChartQA), [original](https://github.com/vis-nlp/ChartQA) | Chart reading and verified arithmetic | Mirror GPL-3.0 label is not an asset-wide guarantee; group chart/table/template variants; underlying tables are verifier-only unless an oracle arm is declared |
| [DocumentVQA mirror](https://huggingface.co/datasets/HuggingFaceM4/DocumentVQA), [original](https://site.docvqa.org/datasets/docvqa) | Document OCR and question answering | Original access/terms and document provenance required; mirror has no blanket card license; group all pages of a document |
| [MathVista](https://huggingface.co/datasets/AI4Math/MathVista) | Evaluation only, 6,141 total examples | CC-BY-SA-4.0 card plus component terms; `testmini` is evaluation, not a training set |
| [MMMU](https://huggingface.co/datasets/MMMU/MMMU) | Multidiscipline visual evaluation only | Apache-2.0 card plus source-image terms; few-shot dev material is not SFT data |
| [RLAIF-V](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset) | Optional later visual preference research, 83,132 pairs | CC-BY-NC-4.0; verify grounding independently; reserve all preferences for an image together and check COCO/TextVQA overlap |

Use Docmatix's row count, image count and QA count as different quantities. Its
pinned `images` metadata reports about 159.4 GB compressed download; the PDF
configuration is separately much larger. These are not sensible default first
downloads on this workstation. FineVision and reasoning collections likewise
require bounded component/shard selection before materialization.

## 4. Additional primary-source candidates

These 12 sources have source links and intended roles, but no immutable dataset
pin in this update. Exact asset terms, version, sizes and checksums must be
established before acquisition/admission. Do not describe them as downloaded.

| Source | Proposed use | Specific control or limitation |
|---|---|---|
| [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/), [generator](https://github.com/facebookresearch/clevr-dataset-gen) | Compositional perception and source-derived verifiers | CC-BY-4.0 dataset; generator separately licensed. Scene graphs/programs stay outside model inputs in perception arms |
| [GQA](https://cs.stanford.edu/people/dorarad/gqa/index.html) | Balanced compositional visual training | Use chosen balanced train split; full question expansions and Visual Genome assets require grouping and term checks |
| [TextVQA](https://textvqa.org/dataset/) | Scene-text reading | Original source located; guessed Hub mirror returned HTTP 401, so no mirror-content verification is claimed |
| [OCRBench / OCRBench v2](https://github.com/Yuliang-Liu/MultimodalOCR) | OCR evaluation only | Choose and pin one version; code license does not settle all assets. Guessed Hub mirror also returned HTTP 401 |
| [LoCoMo](https://github.com/snap-research/locomo) | Longitudinal evaluation only | CC-BY-NC-4.0; only ten conversation clusters, not one independent user per question |
| [LaMP](https://lamp-benchmark.github.io/) | Conditional personalization method training/evaluation | Task/component and noncommercial/share-alike terms; preserve complete user profiles |
| [ScienceQA](https://scienceqa.github.io/) | Optional research visual-science training | CC-BY-NC-SA-4.0; some questions lack images; lectures/solutions are labels, not free inference evidence |
| [Visual Spatial Reasoning](https://github.com/cambridgeltl/visual-spatial-reasoning) | Spatial relations and controlled true/false tasks | Apache-2.0 annotations/code plus COCO assets; random and zero-shot releases share examples |
| [POPE](https://github.com/RUCAIBox/POPE) | Object-hallucination evaluation only | Reserve root images across question regimes and all COCO-derived mixtures; inspect asset terms |
| [IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval) | Text instruction retention evaluation only | Pin executable checks; formatting compliance is not semantic correctness |
| [HumanEval](https://github.com/openai/human-eval) | Code retention evaluation only | MIT repository; sandbox generated code, reserve benchmark solutions |
| [SWE-bench Verified](https://www.swebench.com/SWE-bench/) | Later tool/software evaluation | 500 engineer-confirmed cases; container and repository costs/terms require separate admission; deferred for first vision work |

## 5. Proposed internal data with stronger verifiers

**Fresh temporal exact worlds:** generate complete entity/attribute worlds,
observations, dates, corrections, revoked records, branch scopes and independent
distractors. Ask multiple questions per stored state, including unknowns and
unaffected facts. Hold out complete world/template families and values; reserve
counterfactual siblings together. The canonical ledger computes labels, while
only the permitted observations are supplied to the model or writer.

**Procedural visual chronicles:** render owned scenes, charts, forms and small UI
states with source scene/document truth, then create chronological changes and
delayed questions. Group renderer templates/layouts/scenes before expansion.
Include absent objects/text, changed values, spatial transformations and
unrelated retention. Rendering metadata is verifier-only unless a declared
structured-input control receives it. This proposed corpus is not CLEVR itself
and has not yet been generated.

Synthetic data supplies precise controls but cannot establish broad real-image
competence. Pair it with separately admitted real-image/document evaluation.
Generator bugs require independent tests; a programmatic label is not infallible.

## 6. Proposed staged acquisition and training quantities

These are **planning ceilings and expansion targets**, not approved run protocols,
observed counts or guaranteed convergence. A run must declare actual eligible
counts, tokens, updates, time and numeric quality gates before execution.

| Stage | First bounded tranche | Conditional expansion | Independent evaluation |
|---|---|---|---|
| R1–R3 repairs | Tiny adversarial fixtures; no large dataset intake | Only fixtures needed to expose the contracts | Foreign scope, forged route/status, duplicate updates, representation rebuild |
| V0 interface | Up to 256 eligible/owned assets and 1–2k contract cases | No quality-driven scale-up before the native interface passes | At least 100 source/template families where available; report actual independence and software-only claim |
| V1 alignment | 10k distinct image-caption/grounding pairs | 100k, then 300–500k only after useful held-out image dependence and a measured resource plan | About 1k independent held-out assets with multiple questions and wrong/no-image controls |
| V2 instructions/reasoning | 20k mixed visual examples | 50–100k eligible, balanced examples after the pilot passes | Reserved object/spatial/OCR/chart/document families; independently checked reasoning |
| Text retention replay | Capped 20k-example text pool from allowed components | Refresh by error category, not blanket corpus growth | Broader EN/ZH instructions, reasoning, code, unknowns and tool formats; final audit never becomes replay |
| V3 memory | 1k temporal episodes for the first diagnostic | Up to 10k training episodes; proposed 4–16 observations and 4–8 queries per episode | Target 1k independent held-out families when scaling; corrected and unaffected facts, unknowns, persistence |
| V4 learning | One cycle, 200–1k independently verified practice cases | Ten chronological cycles after the first cycle's quality/retention gates | Parent and strongest practical baseline; unseen later families, no-improvement and rollback periods |

For V2, initially propose **20–30% text replay by supervised tokens**, not example
count. Record image tokens, prompt tokens and response tokens separately; a long
rationale must not quietly dominate the mix. Exact allocations and repeat counts
belong in the prospective protocol. V0 is interface qualification, not a small
training run advertised as a capable VLM.

The first dataset acquisition campaign should have a **25 GiB on-disk asset
ceiling**, selected after checking free space; model checkpoints and derived
artifacts need separate disk estimates. This proposed ceiling does not alter
the existing 50 GiB native RAM guard. Prefer small pinned shards or deterministic
streamed sampling followed by local hashed materialization. Never trigger a
multi-terabyte download through an unspecified default dataset config.

## 7. Deduplication and held-out integrity

Maintain a root-asset registry across sources. Exact byte hashes alone are not
enough: decoded-image/perceptual checks, OCR/document fingerprints and original
IDs detect recompression, crops and mirrored copies. Review ambiguous matches;
record the dedup algorithm/version and false-match samples.

- **COCO family:** LLaVA instructions, VQAv2, VSR, POPE, RLAIF-V and mixture
  components can share images. Resolve one split per root image before questions.
- **Visual Genome family:** GQA and related mixtures share scene/image origins.
  Precomputed features may already contain validation exposure; audit their
  producer and upstream training provenance before treating a split as unseen.
- **Document family:** group all pages, PDFs, crops, OCR and generated questions
  from one document. Docmatix `pdf` and `images` are not independent corpora.
- **Teacher expansion:** all paraphrases, reasoning completions and preferences
  inherit their original question/asset/persona split and lineage.
- **Mixtures:** Cauldron, FineVision, Tulu and SmolTalk are containers; count
  underlying records once. Never use a dataset name as proof of independence.

Keep train, development, calibration, promotion and final audit roles explicit.
Official benchmark splits remain authoritative; a project-level decontamination
filter may further exclude overlaps but may not relabel test data as training.
Report confidence/variation by independent families, not inflated question count.

## 8. Normalized record and verifier contract

Each normalized record needs source/revision/config/split/row ID, root asset and
digest, source-family group, modality, timestamp/cutoff, permitted scope, terms,
producer/processor/crop/layout versions, parent/teacher derivation, input fields,
target fields, target verification state and evaluator version. Memory episodes
add ordered observation/correction IDs and expected eligible view. Keep targets
in evaluator/training-label fields, never embedded into observation payloads.

Use task-appropriate metrics: consensus-aware VQA scoring where required,
document/OCR normalization or official ANLS where specified, chart tolerances,
executable math/code outcomes, and explicit abstention labels. Freeze their
versions and validate false positives/negatives. Token decoding and dataset-derived
gold remain independent of cached model-reported correctness.

Before admission, sample content/rights, reject corrupted assets and unsupported
labels, test missing/duplicate cases, inspect per-source distributions, test all
input paths for answer/future leakage and regenerate the manifest deterministically.
Only then update the runtime dataset registry with exact eligible counts and
artifact hashes for the named experiment. A card hash alone cannot pass this gate.

## 9. First acquisition deliverables

The next agent should prepare a component allowlist, a small eligible V0 asset
manifest, fresh memory fixtures and a capped V1 candidate manifest. It should
record unresolved original-source access or terms and continue with eligible
alternatives. Model weights, bulk corpora and native training remain separate
actions with explicit resource plans. Report acquired bytes and verified rows,
not catalog size, as the amount of usable local training data.

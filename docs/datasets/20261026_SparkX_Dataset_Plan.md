# SparkX 2.6 4B: high-quality dataset strategy

**26 October 2026 · Dataset companion to architecture revision 4.4 · Research shortlist, not training admission.**

The best next step is a curated grounding curriculum, followed by experience-learning episodes. Start with human point/track annotations, controlled acoustic labels, and independently checked audiovisual questions. Large synthetic caption/QA collections can expand coverage after those foundations work. Their size alone is not evidence of quality.

This review covers **36 source families and 42 Hugging Face repositories**, including alternatives and rejected lookups. It inspected pinned cards/metadata and the relevant sections of eight newly retrieved papers, with original-source checks for important releases. The [catalog](20261026_SparkX_Dataset_Catalog.json) contains exact Hugging Face revisions, source roles, annotation provenance, split keys and intake checks. The [research report](../../reports/sparkx_dataset_hunt_20261026/README.md) records discrepancies and the limits of verification. **No image, audio or video assets were acquired or audited; no training was run.** Three small original Clotho provenance CSVs were counted to resolve conflicting metadata.

Use this with the [existing acquisition guide](../08_Dataset_Acquisition_and_Curriculum.md), the [earlier 34-source catalog](20261024_Source_Catalog.json), and the [MiniCPM dataset validation plan](20261026_MiniCPM_Dataset_Validation.md). Earlier text, OCR, document and chart candidates retain their previous review status; this work does not silently re-certify them. The runtime admission registry is unchanged.

## 1. Architectural purpose

**Full offline fine-tuning establishes vision, audio and multimodal reasoning.** Subsequent native learning acquires knowledge and experience using those trained capabilities. Dataset selection must support both stages without conflating them.

| Model component | Required supervision | Strongest starting candidates | What it does not establish |
|---|---|---|---|
| Vision encoder, spatial bridge, evidence heads | Objects, absent objects, points, counting, spatial relations | PixMo-Points; small PixMo-Cap subset; owned scene graphs | General visual reasoning from captions alone |
| Temporal bridge and cross-frame identity | Object trajectories, event locations, ordering, comparisons | Molmo2-VideoPoint/VideoTrack; source-safe multi-image groups | Fine motion at resolutions/FPS not represented in training |
| Native audio pathways | Sound events, pitch/timbre, speech, timing | FSD50K, NSynth, LibriSpeech; controlled audio mixtures | Listening from transcripts alone |
| Audiovisual reasoning | Which sound belongs to which visible event; synchronization, ambiguity, temporal relations | Owned paired AV scenes; reviewed LongVALE research subset; MUSIC-AVQA controls | Causality from co-occurrence captions |
| Semi-cognitive appraisal | Perceived affect distributions, conflicting cues, uncertainty, correction | CREMA-D votes; RAVDESS diagnostic; later consented natural conversation | A speaker's actual private feelings or a model's subjective experience |
| Intrinsic memory and plastic knowledge | Chronological teaching, corrections, source removal, transfer and retention | Fresh owned episode families; selected PersonaMem-v2 train histories | Persistent learning from static QA accuracy |
| Swappable vocabulary | Novel names bound to perceptual concepts, slot turnover, base-token fallback | Fresh labels attached to held-out visual/audio concepts | New concept knowledge from shorter tokenization |
| Unified sensory/action spine | Paired native features, timestamps, coordinate frames and device states | Later task-specific sensor/trajectory data | Device competence from text tool-call datasets |

The [Molmo2 paper, §3.2](https://arxiv.org/html/2601.10611v1) provides direct support for combining captions, pointing and text retention in full-parameter visual training. It does **not** establish that its exact mixture, compute or results will transfer to SparkX. Its visual grounding recipe also does not validate our proposed persistent memory. These remain separate experiments.

## 2. What “high quality” means here

Evaluate quality along separate dimensions: annotation correctness, whether the sensory input is necessary, temporal/spatial precision, source diversity, available provenance/rights, and fit to the intended task. A human label can still be ambiguous; a synthetic label can still be useful when independently verifiable. Neither a popular Hub page nor a model-generated quality score establishes admissibility.

The priorities below are conditional:

- **P1:** first to inspect at asset level because the supervision matches the architecture well.
- **P2:** useful expansion, research-only comparison, or source requiring substantial cleanup.
- **Evaluation:** reserved; never automatically included in replay or online learning.
- **Unqualified mirror:** the original dataset may be valuable, but the inspected mirror does not establish a usable release.

For a product-oriented training set, record annotation permissions, media permissions, redistribution/attribution obligations and intended-use conditions separately. The review reports publisher statements; it does not grant blanket clearance. In particular, an Apache annotation card cannot change the terms of the underlying recordings or captions.

## 3. Vision and spatial/temporal grounding

| Dataset | Evidence and useful signal | Recommended use and important limit |
|---|---|---|
| [PixMo-Points](https://huggingface.co/datasets/allenai/pixmo-points) | Human referring expressions and points; pinned card has 2,376,222 rows. The paper describes about 428k source images | **P1:** point/count/absent-object training. Rows share images. Verify the supplied hash against downloaded image bytes and split by image root |
| [PixMo-Cap](https://huggingface.co/datasets/allenai/pixmo-cap) | 717,042 rows; human spoken descriptions rewritten into long captions by Claude | **P1:** detailed perception/alignment after claim checking. Annotator transcripts are not labels for the sound within an image/video |
| [PixMo-Count](https://huggingface.co/datasets/allenai/pixmo-count) | 36,916 training rows derived from filtered detector outputs; 540 validation and 540 test examples with human-verified counts | **P2:** count expansion. Training points are not human gold; evaluation counts do not provide point-localization ground truth |
| [PixMo-Point-Explanations](https://huggingface.co/datasets/allenai/pixmo-point-explanations) | 79,551 rows linking text and points; publisher calls the format experimental and reports hallucination risk | **P2:** small, reviewed evidence-explanation subset. Prefer concise verifiable evidence over unexamined long rationales |
| [Molmo2-VideoPoint](https://huggingface.co/datasets/allenai/Molmo2-VideoPoint) | 658,340 default-config rows, human point placement, eight categories and `annotator_unsure` | **P1:** temporal/spatial pointing. Start with sure annotations. Preserve raw-frame versus 2 FPS timestamp mappings. Default and category configs are overlapping views |
| [Molmo2-VideoTrack](https://huggingface.co/datasets/allenai/Molmo2-VideoTrack) | Human referring queries over existing tracks; paper reports about 3.6k clips and 15k queries | **P1:** identity persistence and relational tracking. Source videos come from 16 datasets with different terms; keep original video and adjacent clips in one split |
| [Molmo2-MultiImagePoint](https://huggingface.co/datasets/allenai/Molmo2-MultiImagePoint) | Algorithmic groups derived from PixMo-Points | **P2:** cross-image comparison. Its `image_sha256s` are documented as URL-string hashes, unlike PixMo's byte hashes. It is not an independent image corpus |
| [Molmo2-AskModelAnything](https://huggingface.co/datasets/allenai/Molmo2-AskModelAnything) | 129,470 `HumanQA` rows; human/model-assisted open video QA | **P1:** reasoning questions after grounding. Build development splits by source video; `HumanQA` is not a train/test guarantee |
| [Molmo2-Cap](https://huggingface.co/datasets/allenai/Molmo2-Cap) | 104,002 train and 3,815 validation rows; human narration, model rewrites and synthetic frame captions | **P2:** temporal description. Roughly 900-word captions are costly and may reference evidence outside a sampled crop. Source attribution includes restricted third-party corpora |

PixMo/Molmo2 cards declare ODC-BY for their collections, state research/educational intent, and retain underlying media conditions. Some Molmo2 video subsets have CC-BY download mappings; this is not permission for every source. Requester-pays GCS examples can incur charges. Choose a specific eligible source subset and asset manifest before acquisition. These details appear in the [VideoPoint card](https://huggingface.co/datasets/allenai/Molmo2-VideoPoint), [VideoTrack source table](https://huggingface.co/datasets/allenai/Molmo2-VideoTrack), and pinned attribution evidence in the report.

Points and tracks give a concrete route to the requested visual-primitive reasoning: supervise where evidence is, its identity across time, and relations between evidence items. Later train the native reasoning workspace to use those representations. A text answer accompanied by a point is not itself proof of internal visual reasoning; remove or swap relevant regions and test whether conclusions change correctly. Do not generate “not present” negatives from incomplete annotation coverage.

Retain selected chart/OCR/document examples from the earlier catalog so the system does not become a pointing specialist. Select a limited number of verified components from Cauldron/FineVision, not those collections plus all of their constituent datasets. Reserve MathVista/MMMU and their underlying assets. The previous catalog already documents why these are evaluation sources rather than convenient SFT pools.

## 4. Native audio, music and affect

| Dataset | Quality signal and scale | Role and constraint |
|---|---|---|
| [FSD50K mirror](https://huggingface.co/datasets/Fhrozen/FSD50k), [original](https://zenodo.org/records/4060432) | 51,197 human-labeled clips, 108.3 hours, 200 event classes | **P1:** non-speech sound grounding. Clip labels are weak in time and development labels are not exhaustive. Join per-file licenses; CC0/CC-BY subsets are product candidates after checks |
| [NSynth mirror](https://huggingface.co/datasets/jg583/NSynth), [original](https://magenta.tensorflow.org/datasets/nsynth) | 305,979 controlled four-second notes with pitch, instrument and velocity | **P1:** inexpensive acoustic discrimination and compositional controls. Preserve instrument-disjoint training boundary; isolated notes do not cover song structure or natural mixtures |
| [LibriSpeech](https://huggingface.co/datasets/openslr/librispeech_asr) | Read speech and aligned transcripts; `clean/train.100` has 28,539 utterances | **P1:** small speech alignment component. It cannot replace sound/music data or natural conversational prosody. CC-BY card; keep speaker/chapter groups |
| [CREMA-D mirror](https://huggingface.co/datasets/myleslinder/crema-d), [original](https://github.com/CheyneyComputerScience/CREMA-D) | 7,442 acted recordings from 91 actors; separate audio, video and AV listener judgments | **P1:** perceived-affect calibration. Join original vote tables rather than using only filename emotion labels. Original ODbL/Database Contents License; actor-disjoint evaluation |
| [MusicCaps](https://huggingface.co/datasets/google/MusicCaps) | 5,521 musician-written descriptions of ten-second excerpts | **P1, conditional:** musical description and perceptual vocabulary. Preserve AudioSet split membership; 2,858 examples come from AudioSet eval. Captions CC-BY-SA; recording rights separate |
| [Slakh reduced mirror](https://huggingface.co/datasets/DreamyWanderer/Slakh2100-FLAC-Redux-Reduced), [original](http://www.slakh.com/) | Synthesized multitrack music with aligned MIDI/stems; original 2,100 mixtures | **P2:** exact instrument-entry/overlap controls. Original is CC-BY, inspected reduced mirror says NC-SA and has minimal documentation. Prefer verified original release; split by source composition |
| [MusicNet mirror](https://huggingface.co/datasets/DreamyWanderer/MusicNet), [original](https://zenodo.org/records/5120004) | 330 classical recordings, over 1M aligned note labels, musician checks; estimated 4% labeling error | **P2:** real-recording temporal music supervision. HF mirror lacks README. Original release and recording provenance must govern; timing is not exact synthetic truth |

FSD50K's [paper](https://arxiv.org/abs/2010.00475) distinguishes careful human validation from complete event labeling. Use its labels for sound identification, not fabricated start/end timestamps. NSynth/Slakh provide controlled timing but need held-out natural audio to test transfer. MusicCaps supplies rich language but not independently verified chord/key sequences for every excerpt. The sources therefore complement one another rather than substitute for one another.

The following remain valuable **restricted research candidates**, outside the proposed product training pool until rights are separately resolved:

| Dataset | Why keep it on the research list | Restriction or quality limit |
|---|---|---|
| [Clotho mirror](https://huggingface.co/datasets/gijs/clotho), [original v2.1](https://zenodo.org/records/4783391) | Five human audio-only captions per clip; counted 3,839 development, 1,045 validation and 1,045 evaluation metadata records | Original caption LICENSE is experimental/noncommercial; media have individual Freesound licenses. The public page's headline totals conflict with release files |
| [AudioCaps mirrors](https://huggingface.co/datasets/gijs/audiocaps), [official 2.0](https://github.com/cdjkim/audiocaps/tree/master/dataset2.0) | Human sound descriptions; official v2 lists 98,606 caption rows | Academic-use statement; inspected HF mirrors are not verified v2.0. Multiple captions can share one recording. Original protocol allowed contextual assistance |
| [RAVDESS mirror](https://huggingface.co/datasets/narad/ravdess), [original](https://zenodo.org/records/1188976) | Small, controlled speech/song/AV affect diagnostic with validation ratings | NC-SA license; separate commercial license not obtained. HF speech portion is 1,440 files, not all modalities. Acted expressions are not natural-emotion ground truth |
| [WavCaps](https://huggingface.co/datasets/cvssp/WavCaps) | Broad sound coverage and downloadable waveforms; 403,050 weakly captioned clips | Card body restricts academic use despite a CC-BY metadata badge. Descriptions rewritten by ChatGPT can contain facts not audible in the clip |

For the semi-cognitive core, keep three targets distinct: observed acoustic/visual evidence, a distribution over listener-perceived affect, and the model's functional appraisal state. CREMA-D votes support the second. They do not label a speaker's true internal state or the model's own feelings. The [VoxEmo paper](https://arxiv.org/html/2603.08936v1) supports evaluating disagreement and cross-corpus transfer; it is not a verified single Hugging Face waveform dataset. Similar-sounding Hub repository names were not treated as its release. Natural, multilingual and longitudinal affect remain coverage gaps.

## 5. Audio reasoning and joint audiovisual experience

| Dataset | Recommended interpretation | Acquisition decision |
|---|---|---|
| [AudioMCQ](https://huggingface.co/datasets/inclusionAI/AudioMCQ) | 571,118 synthetic QA examples with weak/strong audio-contribution labels; useful methodology for detecting text shortcuts | **P2:** inspect component provenance and source splits. Parent release is annotation-focused; source media separate. Keep bundled benchmark files outside training |
| [AudioMCQ-StrongAC-GeminiCoT](https://huggingface.co/datasets/Harland/AudioMCQ-StrongAC-GeminiCoT) | 19,480 selected examples with Gemini-generated reasoning; official DCASE 2026 training set | **P1, component review:** manageable reasoning seed. Correct teacher answers do not certify every rationale/timestamp. Apache card does not clear inherited media/captions |
| [AudioSkills / XL](https://huggingface.co/datasets/nvidia/AudioSkills) | About 10M synthetic QA over heterogeneous sources | **P2 restricted:** NVIDIA OneWay Noncommercial; annotations without media. Source names are not safe training split definitions |
| [LongAudio / XL](https://huggingface.co/datasets/nvidia/LongAudio) | About 1.25M described QA, 30 seconds–10 minutes, plus newer AFNext files | **P2 restricted:** same license family and external source requirements. Some clips must be reconstructed in exact order; defer until short-audio quality passes |
| [AF-Think](https://huggingface.co/datasets/nvidia/AF-Think) | Current card says 500k; AF3 paper's earlier release said 250k. Overlaps AudioSkills/LongAudio | **P2 restricted:** choose exact revision/file family; do not count parent and rationale variants as independent evidence |
| [MF-Skills](https://huggingface.co/datasets/nvidia/MF-Skills) | Card describes 1.87M QA and 3.39M captions across 2.95M songs, with reasoning files | **P2 restricted:** synthetic music-tool/LALM annotations, external recordings, NC license. Full published row count not independently verified; guessed `nvidia/MF-Think` could not be verified |
| [LongVALE](https://huggingface.co/datasets/ttgeng233/LongVALE) | 8,411 videos, 105,730 timed AV events; automatic training annotations, human-refined test set | **P2 restricted:** strong fit to event-memory interfaces, but NC-SA and quality audit required. Raw videos, features and model checkpoints share the repo; avoid blanket download |
| [MUSIC-AVQA original](https://github.com/GeWu-Lab/MUSIC-AVQA), [HF full mirror](https://huggingface.co/datasets/UnFaZeD07/Music-AVQA), [audit companion](https://huggingface.co/datasets/gwkrsrch/music_avqa) | Musical AV correspondence and temporal QA; useful for modality-ablation controls | **P2 unqualified mirror:** full mirror lacks a card; audit companion is only test indices/probe values. Its MIT license is not a license for the original videos/questions |

The newer [DCASE 2026 report, §2](https://arxiv.org/html/2607.18718v1) adds multiple blind-input filters and human verification to its evaluation construction. This strengthens the case for measuring **audio dependence**, not for adopting every teacher rationale. Its development/evaluation sets also reuse some MMAU/MMAR/MMSU material, so benchmark names are not independent split boundaries. These findings directly inform our MiniCPM controls.

For LongVALE, temporal event labels are useful supervision for a memory write candidate containing an entity, interval, sensory representation and concise evidence summary. Store confidence and provenance with each proposed memory. An automatic caption's inferred cause must not become an unquestioned fact. In initial experiments, independently verify the event before teaching it, then test later retrieval/transfer after the raw clip and its context cache are removed. Full encoder fine-tuning requires raw sensory inputs; released fixed features can support a diagnostic, but cannot stand in for the committed full multimodal training stage.

## 6. Memory, vocabulary and self-learning need new episode data

[PersonaMem-v2](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2) provides declared text/multimodal training splits that can supplement teaching and personalization tasks. [LongMemEval cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) remains evaluation-only. Neither establishes the combined dynamic vocabulary, intrinsic engram and persistent plastic-knowledge mechanism.

Build a **new, owned chronological curriculum** after the perception gate passes:

1. Observe unfamiliar visual objects, timbres, short patterns or AV event rules, with no concept name revealed in filenames or metadata.
2. Receive a teacher's novel name or rule, positive and negative examples, uncertainty and later corrections.
3. Reuse knowledge on new views, recordings or compositions. Change irrelevant appearance/background while preserving the taught concept.
4. Remove old prompts, raw media, source KV, and explicit answer-bearing prototypes as appropriate to the claim under test.
5. Apply interference and vocabulary-slot turnover; retrieve by stable concept identity or express the answer through immutable base-token fallback.
6. Restart the process and evaluate retained learning on independently generated families. For plastic-parameter transfer, retain only the declared learned parameters plus stable weights and reset other answer-bearing state.

Use source-held-out recordings from admitted corpora as stimuli, not their evaluation questions. Split at source recording/instrument/world family **before** generating aliases, mixtures or episodes. All derived captions, sound effects and variants inherit the same split. New labels alone do not make a familiar public example an unseen concept.

This curriculum is a proposal, not an already released dataset or a passed experiment. It should reuse the repository's owned-world discipline and independent verifiers while adding new renderer, acoustic and AV families. The current small renderer is not enough diversity for a product claim. Personality-specific annotation waits for the user's personality-core description; the general memory and uncertainty schema can proceed.

## 7. Evaluation reserve and contamination graph

| Reserved source | Scope | Special handling |
|---|---|---|
| [Molmo2-VideoPointEval](https://huggingface.co/datasets/allenai/Molmo2-VideoPointEval), [VideoTrackEval](https://huggingface.co/datasets/allenai/Molmo2-VideoTrackEval) | Grounding and identity tracking | Hold out whole source videos. Reference masks stay in scoring, not inference features |
| [Video-MME mirror](https://huggingface.co/datasets/lmms-eval/Video-MME) | Broad video understanding | Pin original protocol; report subtitle and native-audio conditions separately |
| [LongVideoBench](https://huggingface.co/datasets/longvideobench/LongVideoBench) | Long video referring reasoning | Gated, NC-SA; no gate accepted. Interleaved subtitles do not prove native listening |
| [MMAU official code](https://github.com/Sakshi113/MMAU), [accessible mirror](https://huggingface.co/datasets/lmms-lab-audio/mmau) | Speech, sound and music reasoning | Verify corrected release version. Author-linked Hub repositories currently contain only metadata files at inspected revisions; mirror parity remains unverified |
| [MMAR](https://huggingface.co/datasets/BoJack/MMAR) | Difficult audio reasoning | Keep original/modified benchmark versions distinct; NC card. Check overlap with DCASE subsets |
| Earlier MathVista/MMMU/LongMemEval/PersonaMem benchmark splits | Retention and existing capability coverage | No replay, fine-tuning or persistent feedback-driven learning from reserved test material |

Construct a provenance graph joining original media, annotations and every derivative. Important families include PixMo → MultiImagePoint; YouTube/AudioSet → AudioCaps/MusicCaps/AudioSkills/AF-Think/AudioMCQ; Freesound → FSD50K/Clotho/WavCaps; source videos → Molmo2/LongVALE/AV benchmarks; AudioMCQ → StrongAC; MMAU/MMAR/MMSU → DCASE subsets. This graph expresses possible overlaps that must be measured, not an assertion that every pair shares every item.

Use exact hashes, source IDs and interval overlap first, then image/audio perceptual matching and song/performance identity where available. Split whole upstream recordings conservatively: two nonoverlapping excerpts may still expose the same event, speaker or song. Report both the number of question rows and unique source groups. A Hub revision pins repository files; it does not pin the bytes later served by an external URL.

## 8. Acquisition order

The next operational step is **a bounded asset audit**, not downloading complete collections. Keep the earlier proposed 25 GiB asset ceiling, subject to an explicit intake manifest and free-space check. The 50 GiB native RAM guard is a different constraint. Model checkpoints, decoded intermediates, audio resampling, feature caches and backups need their own accounting.

| Order | Work package | Suggested first scope, not a run authorization |
|---|---|---|
| 1 | Rights/source map and duplicate graph | Select compatible FSD50K clips, NSynth original split, CREMA-D original votes, and explicit PixMo/Molmo2 media sources; resolve all blocking provenance fields |
| 2 | Asset QA | About 100 distinct source groups per selected source, stratified by task and difficulty; review correctness, unavailable assets and modality necessity |
| 3 | MiniCPM perception/grounding pilot | Up to 10k visual examples and about 15 hours of selected audio; exact allocation in companion plan |
| 4 | AV and memory episode construction | Small source-disjoint AV set and 1k training episodes, with fresh development/audit families |
| 5 | Conditional expansion | 1×/4×/16× curves only after held-out gains, retention and cost gates pass; product full-training scale determined from measurements |

For product data, prioritize independently licensed/owned audio, verified label subsets and human checks. Keep restricted research checkpoints and their descendants out of the product lineage unless their terms are resolved. Do not silently replace a missing source with a different mirror or train a synthetic mixture because it has a convenient loader.

The [validation companion](20261026_MiniCPM_Dataset_Validation.md) specifies candidate budgets, normalized records, modality controls and joint memory/vocabulary/learning experiments. All thresholds and quantities there are prospective design choices; no dataset or model has passed them in this review.

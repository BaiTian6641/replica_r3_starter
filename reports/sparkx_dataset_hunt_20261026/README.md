# SparkX dataset research: findings and verification record

**26 October 2026 · Supports architecture revision 4.4 · Metadata/source research, not an asset or model qualification.**

The review produced a [prioritized dataset plan](../../docs/datasets/20261026_SparkX_Dataset_Plan.md), a [36-family machine-readable catalog](../../docs/datasets/20261026_SparkX_Dataset_Catalog.json), and a [MiniCPM intake/validation curriculum](../../docs/datasets/20261026_MiniCPM_Dataset_Validation.md). It inspected 42 Hugging Face repositories, including alternative mirrors and unrelated or incomplete releases. This is an extension of the earlier repository dataset work, focused on the intrinsic multimodal and learning design.

The most useful additions are human spatial/temporal supervision from PixMo/Molmo2, complementary acoustic labels from FSD50K/NSynth, listener distributions from CREMA-D, and the newer AudioMCQ StrongAC candidate. None is marked training-admitted. Research-only sources remain valuable comparisons but are not treated as product-ready merely because their files are downloadable.

## Evidence package

- [Evidence manifest](evidence.json): 125 retrieval records, 121 successful hashed documents/metadata responses and four unresolved card/API requests. Public search/discovery results are retained as discovery evidence, not used as substitutes for source review.
- [Hugging Face metadata snapshot](hf_metadata.json): exact revisions, card metadata and selected file-inventory facts for 42 repositories. The catalog links to immutable card revisions. Repository revisions do not pin externally hosted media bytes.
- [Clotho metadata audit](clotho_metadata_audit.json): deterministic counts from three original provenance CSVs, including per-file license distributions. These are metadata counts, not decoded-waveform QA.
- [Repository snapshot](repository_snapshot.json): relevant local plan/catalog/registry hashes. Concurrent implementation changes were not modified or regraded.
- [Frozen architecture revision 4.3](architecture_v4_3_snapshot.txt): canonical document before this dataset update.
- [Documentation validation](documentation_validation.json): JSON, links, metadata pins, source hashes and invariant checks. It is not model or dataset quality validation.

Raw retrieved papers/cards are in a temporary research cache; their URLs, retrieval timestamps and SHA-256 digests are in the manifest. They are not copied wholesale into the repository. Exact HF card links remain reproducible while those commits are retained. Original mutable URLs are identified as such; a digest documents the observed content without promising that the URL will serve the same bytes later.

No audio, image or video assets, model weights, or bulk training annotation collections were downloaded. Three small Clotho source/provenance CSVs were retrieved. No third-party dataset loader or model source was executed, no gates or access agreements were accepted, no paid egress was initiated, and the runtime dataset registry was not changed.

## Primary research cross-checks

Eight full paper texts were newly retrieved. The review inspected their data-construction, annotation, split, training and evaluation sections as relevant; it does not claim every line of every appendix was read. Earlier audio/music/affect papers remain in the prior dated evidence packages.

| Paper/source | Grounded finding | Consequence for SparkX |
|---|---|---|
| [Molmo and PixMo](https://arxiv.org/html/2409.17146v1), data/training sections | Human points include exhaustive instances and absent objects; detailed captions use human narration and model rewriting | Separate annotation types. Favor verified points and short evidence targets; do not call rewritten captions wholly human gold |
| [Molmo2](https://arxiv.org/html/2601.10611v1), §§2–3 | Human pointing and tracking complement synthetic QA; full-parameter training combines grounding, captioning and text data | Strong methodological fit for offline native visual training; no direct evidence for SparkX memory transfer or the proposed low-cost budget |
| [AudioMCQ](https://arxiv.org/html/2509.21060v1), §§2–4 | Many audio questions can be answered with silent input; weak/strong partitions are based on a particular model probe | Test signal dependence independently. A “strong” tag is a candidate-selection heuristic, not a correctness guarantee |
| [DCASE 2026 Task 5 summary](https://arxiv.org/html/2607.18718v1), §2 | Evaluation construction combines blind-model filters, text preference/commonsense filters and human checks | Use stronger matched-input controls and retain benchmark lineage. Development/evaluation reused material must be tracked across benchmark names |
| [LongVALE](https://arxiv.org/html/2411.19772v2), §3 and annotation appendix | Event boundaries/captions are automatically generated; held-out test annotations receive manual refinement | Useful event-memory training structure, but automatic train labels need QA. Do not train on the human-refined test set |
| [FSD50K](https://arxiv.org/abs/2010.00475), construction/characterization sections | Human verification improves label quality, while development labels remain incomplete and clip-level | Do not manufacture absent-event or timestamp ground truth from missing tags |
| [Clotho](https://arxiv.org/abs/1910.09387), dataset construction | Multiple captions are collected from audio without auxiliary visual/source hints | Valuable research reference for acoustic grounding; current release rights and counts must come from its own versioned files |
| [AudioCaps](https://aclanthology.org/N19-1011/), annotation protocol | Human annotation uses auxiliary context, so some descriptions can be more specific than sound alone permits | Audit audibility. The original paper does not describe all later v2.0 data, and cannot establish a community mirror's version |

Original Magenta, Zenodo, author GitHub and DCASE pages additionally verified NSynth, CREMA-D, Slakh, MusicNet, RAVDESS, AudioCaps 2.0 and benchmark identities. A primary paper plus a derivative mirror is not enough to establish that the mirror contains exactly the paper's release.

## Discrepancies that changed the plan

1. **Clotho's published totals disagree.** The v2.1 landing page describes 6,974 samples and also quotes development/validation counts one larger than its actual metadata. Directly counted original CSVs contain 3,839/1,045/1,045 records and unique filenames, totaling 5,929 for these three splits. The inspected `gijs/clotho` mirror matches those split counts. Other challenge subsets may exist, but they are not silently added to this release. Two files required explicit non-UTF-8 decoding; raw-byte digests and the extraction repair are recorded.
2. **Clotho is not a blanket permissive caption source.** The original caption LICENSE limits experimental/noncommercial use. Its development metadata contains 1,577 CC0 and 1,572 CC-BY audio entries, but those recording licenses do not change caption rights. Restricted captions remain outside the proposed product lineage.
3. **AudioCaps 2.0 is not the inspected HF mirrors.** The author README lists 91,256/2,475/4,875 caption rows. `gijs/audiocaps` lists 49,431/2,470/4,815 and `OpenSound/AudioCaps` 45,178/2,223/4,411. Do not call the mirrors v2.0 or count all caption rows as distinct audio. Author materials also state academic use.
4. **WavCaps license metadata conflicts with the body.** A CC-BY metadata badge coexists with an explicit academic-use restriction and individual source conditions. The permissive-looking badge was not used to admit the collection.
5. **Human visual data are not uniformly human-authored.** PixMo-Cap is rewritten from human narration; PixMo-Count training labels come from a detector; Point-Explanations carries a hallucination warning. These require different quality checks from PixMo human point placement.
6. **Hash field names do not establish semantics.** PixMo-Points demonstrates hashing downloaded image bytes. Molmo2-MultiImagePoint documents hashes of URL strings. Only independent media hashes can verify content integrity in the latter.
7. **Molmo2 configs and source videos overlap.** VideoPoint's default and eight category configs represent the same collection. MultiImagePoint derives from PixMo. VideoTrack contains mixed source licenses; Cap's attribution file names restricted LLaVA-Video, Koala and InternVid sources.
8. **NVIDIA audio repositories mostly release annotations.** AudioSkills, LongAudio, AF-Think and MF-Skills need source recordings obtained separately. AF-Think's current 500k card and AFNext files differ from the older AF3 250k description. MF-Skills contains reasoning files; a separate guessed `nvidia/MF-Think` API request returned HTTP 401, which does not establish whether that repository exists privately or under another name.
9. **Newer audio reasoning data still inherit source restrictions.** The 19,480 StrongAC card advertises Apache-2.0, but includes sources such as Clotho and AudioCaps. It cannot be assumed to relicense those sources. Card news dates also conflict with the 2026 model/challenge context; use the immutable revision rather than treating those dates as verified release chronology.
10. **Evaluation corrections matter.** AudioMCQ's current card discloses an MMSU scoring bug in released evaluations. This review does not repeat the obsolete headline as evidence of quality. Also, a zero difference between two correctness indicators does not logically imply identical predictions: two different wrong answers also have difference zero. Record full paired predictions, not only that aggregate measure.
11. **MMAU is ambiguous and some author-linked repositories are metadata-only.** `apple/mmau` is a different agent-capability benchmark. The audio project's linked `gamma-lab-umd` repos contain only README/attributes at inspected pins; `lmms-lab-audio/mmau` has 1k/9k data splits but release parity is unverified. The official audio project documents a corrected 2025 release and withheld test answers. A mirror's answer column is not authority to train on or expose hidden test answers.
12. **MUSIC-AVQA audit metadata is not a training corpus.** `gwkrsrch/music_avqa` contains 9,185 test indices/probe values, without question/answer text or video. Its MIT license covers those columns only. The full `UnFaZeD07` mirror has no card. Join audit rows through `source_index`; the source question ID alone is not unique.
13. **Music mirrors can change the apparent release.** Original Slakh is CC-BY; the inspected reduced mirror declares NC-SA and gives almost no transformation details. MusicNet's mirror lacks a README. Original release metadata, splits and source attribution must govern any eventual intake.
14. **Affect file counts overstate independent examples.** RAVDESS's 7,356 files include three modality versions of 2,452 performances. CREMA-D's hard filename labels do not preserve modality-specific listener disagreement; the original vote tables are needed.

These findings are reflected in per-source catalog restrictions and validation checks. They are reasons for targeted curation, not a claim that the underlying research is unusable.

## Practical recommendation

First qualify a small product-candidate stream built from verified original/eligible sources: FSD50K per-file-compatible audio, NSynth, LibriSpeech, CREMA-D with original votes, explicit PixMo/Molmo2 media subsets, and owned AV/learning episodes. MusicCaps and StrongAC deserve early asset/provenance review, but cannot enter wholesale. Use restricted Clotho/AudioCaps/LongVALE/NVIDIA sources only in a separately declared research track unless their terms are resolved.

Then establish perception and spatial/temporal reasoning through full offline training, before evaluating native experience acquisition. The joint test varies intrinsic memory, swappable vocabulary and plastic-parameter learning, includes source removal and restart, and measures useful transfer per total computation. Public QA corpora alone do not test that combined behavior; fresh source-disjoint teaching episodes are still required.

The next uncompleted operational step is asset-level sampling and QA under a named intake manifest. The research task itself is documented and cross-checked; this report does not claim acquisition, broad model competence or learning gains.

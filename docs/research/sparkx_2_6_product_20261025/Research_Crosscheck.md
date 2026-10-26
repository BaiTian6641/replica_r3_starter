# SparkX 2.6 4B research cross-check

25 October 2026. Supporting analysis for the [canonical revision 3 plan](../../architecture/21_SparkX_2_6_4B_Consolidated_Architecture_and_MiniCPM_Validation.md). Paper results below are author-reported, not reproduced locally. The [manifest](../../../reports/sparkx_2_6_product_refinement_20261025/evidence.json) records retrieved versions, raw hashes and the portions actually read. Full text was retrieved where available; “selected sections read” is deliberately different from a cover-to-cover review or code reproduction.

## Findings that change the architecture

The strongest synthesis is a compact multimodal backbone with explicitly grounded state, a native bounded record bank, and separate offline and online learning regimes. There is support for each ingredient. There is **no published result among the inspected sources that validates this entire combination on Spark 4B**, including recycled vocabulary identities, audiovisual memory, online skill updates and recursive improvement.

Three distinctions prevent overclaiming: a model accepting images is not evidence that its intermediate visual state matters; audio transcription does not preserve every vocal cue; a fixed memory budget does not mean low total processing cost. These distinctions directly inform the validation controls.

### R1. Thinking with Visual Grounding — closely matched 4B evidence

[Paper, v1](https://arxiv.org/html/2606.16122v1); [author code, pinned](https://github.com/Jun-Kai-Zhang/visually_grounded_thinking/tree/b76028f04e04fe241569b4c029467932bdf92588). Read §§3–5 and appendices A/B; inspected repository README.

Gemma3-4B-IT is trained on parallel grounded/non-grounded traces, with point/box supervision and grounding-aware GRPO. The final data has 19,909 paired rows. Box-reward average accuracy on MultihopSpatial is 37.68 versus the base model's 22.70; counting remains difficult. The non-grounded reasoning control suffers rollout-length collapse, which complicates attribution solely to geometric evidence. Four stochastic inference passes and pass@4 are reported separately. The total study costs about 400 H200 GPU-hours.

**Decision:** reproduce grounded primitives as a strong control. Do not import the cost, pass@4 or narrow benchmark superiority as a product-level 4B guarantee. Require matched decode budgets and causal evidence-use tests.

### R2. Latent Visual Reasoning — continuous-state feasibility

[Paper, v1](https://arxiv.org/html/2509.24251v1). Read §3, objectives and decoding discussion.

LVR supervises hidden states against query-relevant visual features, interleaves latent and textual generation, and adapts GRPO while scoring text outputs. Region supervision is available during training, not supplied as an inference oracle. Hidden-state feedback is a concrete native mechanism, but each latent generation step still uses the backbone. Its reward path is not an ordinary categorical policy distribution over continuous vectors.

**Decision:** test short, supervised visual-state recurrence. Do not label feature reconstruction, additional tokens or generated coordinates sufficient evidence of reasoning. Keep final-answer and representation dependence as separate measurements.

### R3. CoVT — compact visual supervision without resident expert tools

[Paper, v1](https://arxiv.org/html/2511.19418v1). Read §3, training pipeline and main ablations.

CoVT trains continuous visual representations using segmentation, depth, edge and DINO supervision, with trainable projections and a LoRA-adapted VLM. The inference path can omit dense expert decoding. The evidence supports teacher-supervised visual representations without requiring a pixel generator during reasoning. It does not establish temporal tracking, small-model lifelong learning or that all expert targets are necessary.

**Decision:** begin with geometry and region targets already available in our data; add a single useful teacher objective at a time. Compare a small recurrent state against an equally expensive plain module, and measure whether removing the learned visual state damages the relevant task.

### R4. Imagination Helps Visual Reasoning, But Not Yet in Latent Space — counterevidence

[Paper, v1](https://arxiv.org/html/2602.22766v1). Read §3's input/latent interventions and probing, plus §4 and main comparison.

The authors analyze Monet, LVR and Mirage representations. Their general input analysis uses 100 instances; detailed answer interventions mainly target Monet and Mirage. Replacing latent contents often changes final answers little. The critique therefore challenges specific implementations, not every possible continuous-state method. Large unnatural interventions can also induce repetition, so damage alone is not proof of meaningful reasoning.

**Decision:** require answer-changing image pairs, matched-state swaps and nuisance interventions. Test both input→state sensitivity and state→answer usefulness, including a sufficient-state task with original visual KV unavailable. Keep an explicit grounded/textual reasoning control.

### R5. DeepSeek “Thinking with Visual Primitives” — unresolved primary provenance

The named [official repository](https://github.com/deepseek-ai/Thinking-with-Visual-Primitives) and its [GitHub API endpoint](https://api.github.com/repos/deepseek-ai/Thinking-with-Visual-Primitives) returned 404 in this audit. Searches found third-party copies, but no accessible authoritative version that could authenticate their contents. This does not prove the work never existed or why access changed.

**Decision:** preserve the user's intended point/box reasoning direction, supported by R1–R3. Do not repeat the mirrors' DeepSeek benchmark, architectural or compression claims as verified facts. Revisit if an author-controlled report becomes available.

## Audio and audiovisual reasoning

### R6. SALMONN — native hearing beyond transcription

[Paper, v2](https://arxiv.org/html/2310.13289v2); [author repository](https://github.com/bytedance/SALMONN). Read §3 architecture and training rationale.

SALMONN combines frozen Whisper and BEATs features, a window-level Q-Former and an adapted language model. Its paired encoders address complementary speech and non-speech information. The work also describes task-overfitting and activation tuning: strong performance on trained audio tasks does not ensure flexible audio instruction following.

**Decision:** use a dual-encoder implementation only as an optional reference/teacher. The product should earn equivalent task coverage with one compact encoder rather than automatically carrying both towers.

### R7. SALMONN-2 — multi-level features and lexical audio context

[Paper, v1](https://arxiv.org/html/2607.17079v1); [author release](https://github.com/bytedance/SALMONN/tree/d96495623580e14bfc1a5204e2852bc1e8007d53). Read §§III–V and training/configuration tables.

The system uses a frozen 600M SPEAR XLarge encoder, multi-layer fusion, 10 Hz audio embeddings and LoRA. About 18.2k hours of instruction data are reported. Explicit contextual-ASR training includes pronunciations, distractors and target dropout; merely adding a hotword list is not consistently helpful. Timestamp injection aids temporal localization without expanding the permanent tokenizer.

**Decision:** connect pronunciation exemplars to lexical memory, with distractors and missing-target controls. Test selected-layer fusion at small scale. Shrinking the encoder to SPEAR Base and using a narrow bridge is our hypothesis, not the published configuration.

### R8. SPEAR and its compact checkpoint — a practical unified candidate

[Paper, v1](https://arxiv.org/html/2510.25955v1); [updated Base checkpoint](https://huggingface.co/marcoyang/spear-base-speech-audio-v2/tree/7447901d44bf265b3c304cc4ccf25ca741527b6c). Read objectives, §4 configuration and §5 evaluation; inspected both Base cards/configs and first-version wrapper/model source.

SPEAR learns from complementary speech/audio teachers. The paper includes Base, Large and XLarge, so the 600M SALMONN-2 front end is not the only possible size. The updated Base release retains 93,289,476 serialized parameters and non-causal configuration; its card describes token-mixing for overlapped/noisy speech. Its speech pretraining is English, and it uses custom Zipformer code.

**Decision:** favor it for unified-audio experiments, with Whisper-small as the multilingual speech control. Native MPS kernels, streaming conversion, feature-layer access and licensing of every copied dependency still need ordinary implementation qualification. No weights were downloaded or code executed here.

### R9. MiniCPM-o 4.5 — concrete integration example

[Official architecture documentation](https://openbmb.github.io/MiniCPM-o-Demo/site/en/model.html); [pinned model source](https://huggingface.co/openbmb/MiniCPM-o-4_5/tree/eb4141de24550a3a1a4082f718be58f46b761728). Read architecture/pooling documentation and inspected config plus audio-embedding code.

The inspected release uses a Whisper-medium-derived audio encoder, a projection and pooling step five. Source code has explicit chunk-attention handling; this is more than adding audio tokens to a text prompt. It is a roughly 9.37B complete multimodal checkpoint, not the text-only MiniCPM5-2B used in our experiments.

**Decision:** reuse interface ideas, not checkpoint identity or performance claims. No additional speech decoder is needed for our audio-input/text-output milestone. Validate padding, chunk boundaries, lookahead and clock alignment explicitly.

### R10. Qwen3-Omni — joint training and its scale boundary

[Paper, v1](https://arxiv.org/html/2509.17765v1). Read §2's encoders/time alignment and §3 training stages.

The report describes AuT at roughly 650M parameters, 12.5 Hz audio inputs, time-aligned audiovisual representations and staged multimodal training including full unfreezing. AuT uses 20 million hours of supervised audio; the multimodal training scale is vastly beyond this project. Its custom multimodal positional encoding and serving latency cannot be transplanted as Spark properties.

**Decision:** keep a common media clock and mixed-modality retention training, while preserving Spark's existing positional scheme initially. Low audio token rate is a measurable design variable, not a promised end-to-end latency. Qwen3-ASR's separately retrieved report was background discovery, not evidence that its whole model is a small standalone encoder.

### R11. MOSS-Audio — inter-layer acoustic evidence

[Paper, v1](https://arxiv.org/html/2606.01802v1). Read §2's encoder, DeepStack-style injection and timestamps.

MOSS-Audio uses a roughly 0.6B encoder, local acoustic attention, 12.5 Hz output and an additional intermediate-feature injection path. This supports the motivation to retain lower/middle-level cues alongside the final semantic output. It does not determine which two Spark depths are optimal or prove that adding several ports is cost-effective.

**Decision:** compare selected-layer input fusion first; only add acoustic residual injection if it improves event/prosody reasoning over that cheaper control. Share existing model ports and profile the cost of materializing intermediate encoder states.

### R12. video-SALMONN-o1 — training for audiovisual reasoning

[Paper, v1](https://arxiv.org/html/2502.11775v1); [ICML proceedings](https://proceedings.mlr.press/v267/sun25x.html). Read task/data description and process-preference rationale; the full optimization implementation was not audited.

The work combines reasoning-intensive AV examples with contrastive step selection for process DPO. This supplies a concrete training direction beyond captioning or ASR. Its reported gains involve a different multimodal backbone and training set; more elaborate reasoning traces alone do not demonstrate acoustic dependence.

**Decision:** build audiovisual cases with verified temporal evidence, use correctness/grounding supervision first, and consider process preferences after the perception path works. Require audio-only, video-only and time-shifted controls for claimed cross-modal reasoning.

### R13. video-SALMONN S — memory links directly to the product target

[Paper, v1](https://arxiv.org/html/2510.11129v1). Read §§3–5 and optimizer/memory-size design discussion.

The visual stream passes through test-time-trained memory before fixed-budget token selection; audio bypasses the fast layer. The implementation uses Hessian-free updates, normally three conjugate-gradient iterations, and prompt-dependent reading. Reported AV training takes 48 hours on 32 H800s; inference uses an 80GB H800. Larger memory settings and question-conditioned reading still incur computation.

**Decision:** it supports studying streaming memory, not a low-cost Mac reproduction. Start with bank-only and a small linear/SGDN-2 control, then compare nonlinear TTT at equal bytes and measured work. Shared visual/audio memory requires new evidence.

## Affect, appraisal and evaluation

### R14. emotion2vec — trainable vocal representations

[Paper, v1](https://arxiv.org/html/2312.15185v1). Read §§3–4, frame/utterance losses and frozen downstream setup.

The method combines masked frame objectives and utterance-level distillation with an EMA teacher. Frozen representations support lightweight downstream tasks. “Online” here refers to teacher/student training updates; it is not an evaluated persistent conversational memory or learning-policy system.

**Decision:** distill useful vocal features into the shared audio representation if they improve held-out performance. Keep appraisal separate from emotion labels and preserve uncertainty rather than treating a classifier output as a person's internal state.

### R15. VoxEmo — disagreement is part of the target

[Paper, v1](https://arxiv.org/html/2603.08936v1). Read benchmark design, label distributions, protocol and limitations.

The benchmark spans 35 corpora and 15 languages. Soft-label analysis is restricted to five corpora with annotator metadata, rather than all 35. Perceived and expressed emotion labels differ, and prompts affect results.

**Decision:** report macro-F1/unweighted recall plus distributional calibration where labels permit. Split by speaker/corpus, preserve listener disagreement, and do not compare models on different clip subsets as a single leaderboard. Measure evidence-grounded interaction separately from emotion classification.

### R16. Synthesized-speech emotion study — a necessary distribution-shift check

[Paper, v1](https://arxiv.org/html/2603.16483v1). Read §§3–5 and evaluation/annotation discussion.

The authors find substantial human-versus-synthetic speech generalization gaps and semantic shortcuts in generative emotion inference. Adaptation can reduce some gaps without establishing robust cross-generator transfer. A label requested from a synthesizer is not necessarily the affect human listeners perceive.

**Decision:** synthetic speech may expand training, but final affect tests require natural recordings and independent listeners. Hold out synthesizer families when testing generated audio; contrast identical transcripts with different delivery. Do not use the same emotion model as teacher, reward and sole evaluator.

### R17. SpeechEQ and EchoMind — interaction beyond a single label

[SpeechEQ, v1](https://arxiv.org/html/2606.25990v1), methods/results read; [EchoMind conference abstract](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5b0f168359c170e6c8d2d7be9a713be4-Abstract-Conference.html), abstract only.

SpeechEQ uses multi-turn tone-dependent situations and compares cascaded and native systems. Its human correlation check uses only 15 examples and six model outputs, so a headline score should not be treated as comprehensive emotional intelligence. EchoMind's abstract describes linked content, vocal-cue, reasoning and response tasks using neutral scripts; its full methods were not verified here.

**Decision:** build linked tasks with same-word/different-prosody controls, delayed context and useful response choices. This motivates evaluation structure; neither source proves simulated feelings or personality fidelity.

### R18. Audio-judge shortcuts — independent verification must be tested

[Paper, v1](https://arxiv.org/html/2607.13477v1). Read §III protocols and result tables.

The study manipulates classifier labels, reference placement and A/B order. Several judges follow wrong textual hints despite audio being available. A successful transcription/language test does not establish equal reliability on affect judgments.

**Decision:** test wrong-hint resistance, response-order swaps and parse failures. Verification for self-learning must use task-grounded outcomes and human review where needed; agreement between two similarly biased models is insufficient.

## Earlier memory/vocabulary/LoRA findings retained

The [dynamic-memory report](../dynamic_engram_20261025/README.md) and [vocabulary/LoRA report](../vocabulary_lora_20261025/README.md) remain relevant. MemT, Titans and HOPE motivate native adaptation; static Engram supplies a different pattern-capacity mechanism. DVA/DVAGen supports trained dynamic rows, but our recycled slots, tiny phrase encoder and state migration remain new engineering/research requirements. SEAL supports evaluating useful self-edits, not self-certifying knowledge.

For online skill updates retain tuned ordinary LoRA as the reference; compare LoRA-FA with its actual optimizer correction, then optional rsLoRA/LoRA+, EVA/PiSSA or CARE/Matryoshka only at matched data and optimizer-search budgets. These alternatives optimize different costs or initialization properties. None replaces the user's required full Spark backbone training. Full-training optimizer choice, online PEFT choice and MoE expert-aware adapters are separate decisions.

## Integration risks and falsifiers

| Risk | Evidence that would reject the proposed combination |
|---|---|
| Visual latent bypass | Correct answers persist when relevant latent state is replaced, with no advantage over matched explicit reasoning |
| Audio semantic shortcut | Same words/different sound cannot change decisions; no gain over transcript-only control on cue-dependent tasks |
| Memory interference | Retention improves only when an evaluator refreshes the known future target |
| Vocabulary identity corruption | Slot turnover changes old interpretations, cached inputs or replay targets |
| Representation drift | Full FT makes old bank/phrase vectors unusable without a measured migration strategy |
| False self-learning | Gains vanish on fresh entities/rules or depend on retaining answers in prompt/KV |
| False RSI | Policy revisions consume more search/replay without greater future utility at equal budget |
| Hidden high cost | Encoder/TTT/replay time or global KV grows past the declared long-session budget |

The combination is worth testing because these mechanisms address different bottlenecks. It should be retained only if the controlled integrated model beats simpler, equally resourced alternatives.

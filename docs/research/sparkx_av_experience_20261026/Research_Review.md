# Research review: visual reasoning and reusable audiovisual experience

**26 October 2026.** This review uses paper methods, evaluation protocols and selected code, rather than treating search summaries as proof. Reported results remain author-reported; no paper was reproduced here. Reading scopes and hashes are in the [manifest](../../../reports/sparkx_av_experience_20261026/evidence.json). The proposed synthesis is developed separately in the [design](Native_Experience_Design.md).

## 1. Thinking with Visual Primitives: useful mechanism, different scale

The [mirrored report](https://www.k-a.in/Thinking_with_Visual_Primitives.pdf) interleaves points and boxes with reasoning so references remain spatially explicit. Its backbone is **DeepSeek-V4-Flash, 284B total/13B active parameters**, not a small dense model. At 756×756, patching yields 2916 patches, spatial merging yields 324 language-model positions, and compressed attention yields 81 visual KV entries. These are three different counts; the last is not the prefill length.

Training includes grounding, separate point/box specialists, reinforcement learning, consolidation, and on-policy distillation using student trajectories with reverse KL. The limitations include explicit prompting and imperfect cross-scene generalization. It supports investigating grounded references; it does not establish Spark efficiency, persistent memory or online learning.

Two [report copies](https://raw.githubusercontent.com/xuxinping/Thinking-with-Visual-Primitives/main/Thinking_with_Visual_Primitives.pdf) have SHA-256 `1951785873385498b608b7aa66868f9915ecd78f6c578d42ef52d80ad25f7153`. Matching bytes resolve content consistency only. Original-release provenance and benchmark reproducibility remain unverified.

**Design decision:** implement small grounded heads and causal reference tests first. Do not port compressed sparse attention or inherit the report's headline quality from coordinate tokens alone.

### A public 2B example is not a faithful reproduction

The [unofficial PyTorch implementation](https://github.com/vra/Thinking-with-Visual-Primitives-pytorch/tree/b4ba3251b501985cbf15226bb15f16d12b39cd8f) provides Qwen2-VL-2B/LoRA examples. Static inspection found two consequential differences:

- In [the model wrapper](https://github.com/vra/Thinking-with-Visual-Primitives-pytorch/blob/b4ba3251b501985cbf15226bb15f16d12b39cd8f/model/vl_model.py), optional spatial compression is constructed, but the inspected `forward`/`generate` delegate directly to the VLM without invoking it.
- The [named OPD loop](https://github.com/vra/Thinking-with-Visual-Primitives-pytorch/blob/b4ba3251b501985cbf15226bb15f16d12b39cd8f/unified/train_opd.py) uses dataset batches and teacher-forced distributions. Its teacher-weighted cross-entropy is not the report's student-rollout reverse-KL objective.

This is useful implementation scaffolding, not a qualified 2B replication. No code was executed. Any borrowed implementation must demonstrate the intended forward path, objective direction, rollout distribution and controlled evaluation.

## 2. Spatial and temporal reasoning mechanisms

| Source and inspected scope | Mechanism and evidence boundary | Consequence for Spark |
|---|---|---|
| [VideoLLaMB, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_VideoLLaMB_Long_Streaming_Video_Understanding_with_Recurrent_Memory_Bridges_ICCV_2025_paper.pdf), §§2.2–2.3, 3.1, 3.7–3.8 | Recurrent memory bridges operate between vision encoder and LLM. The ablations favor retaining current visual features alongside memory. Its historical cache grows with segments: the paper describes O(K) storage and O(K²) total retrieval work. Vicuna-7B experiments do not establish compact-device lifelong operation. | Test a small recurrent bridge before the backbone, but replace growing history with a bounded bank and preserve a recent perceptual window. |
| [LatentVC](https://arxiv.org/html/2607.02607v1), method and implementation details | Qwen3.5-9B-based latent visual reasoning aligns latent states with keyframes and then applies vision-oriented reinforcement learning. Previous hidden states can become subsequent latent inputs. The original video prefix remains available; this is not by itself persistent episodic memory. Each autoregressive latent step still runs the backbone. | Treat continuous reasoning as an optional, metered branch. Distinguish reasoning over live evidence from recall after that evidence is removed. |
| [Q-GeoMem](https://arxiv.org/html/2605.27318v1), §§3–4.1 | Combines recent detail with geometry-aware memory, camera information and a reliability gate. Its persistent selection uses the question; geometry comes from an additional frozen CUT3R tower. Experiments use a 7B VLM and substantial training infrastructure. | Useful geometry/reliability ideas, but unknown-future-query writing requires a different protocol. Do not promote relative image coordinates to world coordinates. |
| [ObjectStream](https://arxiv.org/html/2607.28312v1), method/setup/ablations | Training-free object anchors organize histories, changes and recent context from a frozen VLM. Online examples use 1 FPS, at most 256 frames and a 64-object ×8-token history allocation. It reports improved video benchmarks, but does not train native memory writes or durable model skills. | Use object/event continuity as an organizing principle and a baseline. Bound auxiliary history pools as well as the selected output; benchmark beyond short clips. |

LatentVC's [author code](https://github.com/BRZ911/Latent-VC/tree/3e3d3e08df1a051b20e5e17a6c7792e4d52bb7bb) was inspected selectively. Its SFT forward exposes conditional hidden-state feedback and a keyframe alignment loss. The complete recurrent training/inference call chain was not qualified. A configuration label or a hook's presence is insufficient evidence of recurrence being used in the measured path.

## 3. What a useful memory retains

**MemoryVLA.** The [paper](https://arxiv.org/html/2508.19236v1), §§3.2–3.4 and implementation appendix, separates perceptual and semantic memories, retrieves them with gated attention, and compresses adjacent similar entries. This provides a concrete precedent for retaining sensory detail alongside abstract context. Its system drives a robotic diffusion action expert; its memory “consolidation” merges representations and is not evidence of post-deployment parameter learning. Spark should test a paired record against semantic-only storage, while preventing averaging across contradictory state changes.

**MemoryVLA++.** The [extension](https://arxiv.org/html/2606.09827v1), world-model subsection, adds a roughly 1.5B video diffusion component to supply imagined future features, with compression and gated memory use. This is informative for anticipation, but an extra generative world model is not a low-cost default for Spark. Our alternative is a small predictor trained on observed future evidence. Hypothetical features must remain distinguishable from observations and cannot certify their own correctness.

**VisMem.** The [revised paper](https://arxiv.org/html/2511.11007v2), §§3.1–3.4, trains short/long memory formation and a learned invocation policy. Special tokens trigger insertion of compact continuous memory, with experiments over several memory lengths. It supports learning when to use additional visual state. These “memories” do not automatically constitute a writable, persistent experience bank; formation can also involve extra model passes. We therefore meter formation, invocation and reading separately, and test persistence after source removal and process restart.

**LatentStream.** The [paper](https://arxiv.org/html/2609.04131v1), §§3.2–4.1, uses a frozen VLM, a bounded visual bank and test-time optimization of latent states, with an entropy-based objective. It is adaptation of inference state, not durable backbone training. Increased confidence is not independent evidence of increased correctness. This is a useful comparison for latent-state adaptation, but not a sufficient criterion for promoting a learned visual/audio rule into persistent skills.

## 4. Revised video-SALMONN S changes the relevant comparison

The [final paper](https://proceedings.mlr.press/v306/sun26s.html) and [arXiv v2](https://arxiv.org/html/2510.11129v2), §3 and appendices E/F, use **TTT_MEM with one SGD step**, reconstruction and a long-span objective involving past features. Revision 3's v1 review instead discussed Hessian-free conjugate gradients; those descriptions must not be mixed.

Audio bypasses the fast-weight layer. The model also has a bounded token bank and prompt-dependent reading. Adding memory at every tested deepstack level did not improve the reported comparison and increased overhead. The method does not justify duplicating fast weights at every Spark layer.

Its ELViM task replays a prefix of a previously seen procedural video before asking what follows. That supports an episodic-use test, not generalization to a new rule family. For Spark, compare a bounded visual/audio fast-state option against the bank, and separate replayed-prefix recall from new-instance transfer. The long-span target concerns past information; it must not be described as a future prediction loss.

## 5. Audio and vision can reinforce learning, with important qualifications

**MJEPA.** The [paper](https://arxiv.org/html/2606.25225v1), §§4.1–4.4 and ablations, predicts masked and cross-modal latent targets from an EMA encoder. Sharing an encoder alone degrades unimodal representations in its controlled comparison; explicit cross-modal prediction produces positive transfer. High-level cross-modal targets are pooled, rather than assuming one audio patch corresponds to one image patch. The results involve much larger encoders and training than our compact towers. Our proposed separate-tower/shared-predictor variant is a new experiment, not a reproduction. Masked self-supervision also does not automatically enforce causal streaming.

**Enabling Chatbots with Eyes and Ears.** The [ACL paper](https://aclanthology.org/2025.acl-long.1519/), §4.2 and training appendix, demonstrates a relevant small-model example: Qwen2-VL-2B with rank-8 LoRA, followed by frozen CLAP audio features and an adapter, plus multimodal conversational retrieval. The M3C conversations are synthesized; retrieval is an external embedding-memory pipeline. This supports feasibility of audio-conditioned small models and multimodal conversational memory, but does not establish native backbone memory, precise temporal acoustics or persistent skill learning.

**Spark implication:** store acoustic evidence as well as semantics, align modalities with timestamps and uncertainty, and train cross-modal tasks with missing/conflicting inputs. A speech transcript cannot establish that the model retained timbre, sound order, rhythm or perceived prosody. The prior compact SPEAR/Whisper comparison and affect-calibration requirements remain necessary.

## 6. Bounded streaming and retention during full fine-tuning

**StreamingVLM.** The [revised paper](https://arxiv.org/html/2510.09608v2), streaming inference/training sections, uses different retention for sinks, text and video, and trains on overlapping interleaved chunks. This is evidence that the training context should resemble the deployment eviction policy. It is not an episodic database or persistent-learning mechanism. Reported H100 throughput cannot be transferred to MPS. Spark's custom sliding/full attention and positional behavior require independent qualification before adopting cache-position transformations; a trained segment-reset control is the simpler starting point.

**Continual VLM forgetting study.** The [PMLR paper](https://proceedings.mlr.press/v330/srivastava26a.html), §§3.2, 4.2–4.3 and discussion, measures language forgetting while adding vision across multiple small-model sizes. Its “SoftTargets” means **label smoothing**, not distillation from an old frozen language model. Its continual comparison combines replay with a method containing LoRA. This motivates retention data and a label-smoothing ablation, but does not establish an all-parameter Spark recipe. Full-backbone training must retain its explicit parameter scope and report language, vision and audio retention independently.

## 7. Cross-verification and resulting confidence

| Proposition | Assessment |
|---|---|
| Spatial references can make visual evidence less ambiguous | Supported enough to prioritize an experiment; precise causal improvement on Spark remains unmeasured |
| Temporal identities and a recent window complement compressed history | Supported across different systems; our learned bounded implementation still needs testing |
| Sensory evidence and semantic summaries are complementary | Supported design direction; proposed tiny per-record vectors have no established quality guarantee |
| Native visual fast memory automatically learns audio | Unsupported; revised video-SALMONN S explicitly bypasses audio in that path |
| Confidence-driven latent updates establish self-improvement | Unsupported without independent accuracy, persistence and retention tests |
| An efficient multimodal model can learn from experience forever at constant cost | Too strong: finite capacity, encoder cost, replay, verification and forgetting remain unavoidable |
| A 4B backbone can integrate the proposed system through full FT | Plausible engineering hypothesis; local MiniCPM results and current rejected memory/vision runs do not qualify it |

The strongest combined direction is **grounded references → temporal evidence → bounded native records → learned retrieval → verified adaptation**. The contribution we still need to establish is that training this chain together improves new-task performance within the product's resource budget. No single source demonstrates that complete chain at our scale.

# Research cross-check: native learning, audio perception and neural interfaces

26 October 2026. Selected methods, experiments and limitations were read from eight new full-text papers and three previously downloaded foundations. Two official implementations were inspected as source, without importing or executing them. [Evidence manifest](../../../reports/sparkx_intrinsic_learning_spine_20261026/evidence.json) records URLs, versions, hashes and reading scope. This is a focused design review, not an exhaustive literature survey or experimental reproduction.

## 1. The conclusion supported by the evidence

**A frozen stable backbone can belong to a model that continues learning.** The complete model must contain a changing, persistent substrate: associative matrices, plastic MLPs, learned address parameters, explicit records, or a combination. Calling the original 4B weights frozen does not make these other parameters disappear. Conversely, changing a tensor does not establish useful learning: new-instance transfer, correction, retention and causal dependence on the changed substrate must be measured.

The proposed combination is our synthesis. No reviewed paper demonstrates the complete Spark-sized system with visual reasoning, music/affect perception, persistent instructed learning, vocabulary swaps and device interfaces together.

## 2. Native memory and learning machinery

### MemT-1.0: the closest architectural foundation

[MemT-1.0 v2](https://arxiv.org/html/2610.26825v2), §§2–4 and 7, separates learned memory machinery from trajectory-specific state. During serving, backbone and hyper-memory weights stay fixed; local matrices change. Reads use the incoming snapshot, with storage after a turn/segment. Its native branch joins attention before the backbone output projection. This supports permanent memory blocks in a modified decoder. Its coding experiment internalizes retrieved procedural experience; external experience retrieval remains part of that application. It does not establish autonomous lifelong skill acquisition or audiovisual learning. The reported 34B-token training program also rules out treating connector attachment alone as evidence of an inexpensive training recipe.

**Source cross-check.** At official commit [`e75c13d090a6a9d34b29146d901bd84323185f24`](https://github.com/MemTensor/MemT-1.0/tree/e75c13d090a6a9d34b29146d901bd84323185f24), `QwenMemoryBlock.packed` has a read-only path and a commit path returning new state. `QwenMemoryHooks` adds memory before `o_proj`; the released serving integration uses hooks. `MemTSession.load_state` clears plaintext turns and enables memory reads. These source observations support integration and persistence mechanics, not quality reproduction. Spark should use its explicit native executor and its own attention geometry.

### TTT-E2E: learn the initialization for subsequent learning

[TTT-E2E v2](https://arxiv.org/html/2512.23675v2), §§2.2–2.4 and 3.7, optimizes an outer initialization for performance after inner updates. Its practical design changes MLP weights in the last quarter of blocks while retaining a static MLP; other inner-loop weights stay fixed. The main long-context setting uses an 8K attention window and 1K update batches. The paper reports substantial short-context meta-training overhead, and its 3B results require large-scale training. It supports a plastic subnetwork and an explicit cost trade-off, not an established cheap retrofit to Spark or a months-long retention result.

At official commit [`a4fc4788ace38e29b5067916d4f4be33da894085`](https://github.com/test-time-training/e2e/tree/a4fc4788ace38e29b5067916d4f4be33da894085), `inner_parameters` selects suffix parameters, and `loss_for_sequence` initializes a sequence-local inner optimizer and works on copies. Persisting learning across independent sessions is therefore an additional design decision. The implementation is JAX/GPU-oriented; its performance is not an MPS measurement.

### HOPE / Nested Learning: useful organization, qualified claims

[Nested Learning / HOPE](https://arxiv.org/html/2512.24695v1), §§7–9, supplies a vocabulary for interacting update processes and memory with different update frequencies. It motivates separating recent associations, slower knowledge and the learning machinery itself. It does not justify calling several manually chosen timers a learned learning hierarchy. The final self-referential derivation keeps its input query projection fixed even though other quantities adapt; it is not blanket evidence that every address projection learns online. Our feedback-updated address transform and persistent plastic skill bank remain experiments. Self-produced targets can train representation consistency, but do not independently establish factual correctness.

### ANML: a direct, limited precedent for a frozen controller and plastic learner

[Learning to Continually Learn v2](https://arxiv.org/pdf/2002.09571v2), §3 and Algorithm 2, meta-learns a gating network and a prediction network. At meta-test time the gating network stays fixed and only the prediction network's final fully connected layer is updated. The evaluation learns up to 600 unseen Omniglot classes sequentially, using 15 examples per class and testing five different examples. There is no reset between those classes. This is concrete evidence that a stable controller can support persistent learning in another parameter subset. It is a character-classification result, not a language, audio, open-world or subjective-emotion result. Use it to motivate selective plasticity controls, not to claim catastrophic forgetting is solved.

## 3. Audio needs acoustic evidence, temporal structure and suitable supervision

### Audio Flamingo 3

[AF3 v2](https://arxiv.org/html/2507.08128v2), §§3–6, starts from a Whisper large-v3 encoder trained for speech, sound and music descriptions, with a Qwen2.5-7B language model. Its pipeline includes connector alignment, encoder tuning and full fine-tuning. The context/thinking stage uses LoRA; the chat stage again fine-tunes the model. This is not evidence that all stages are full fine-tuning or that a 93M encoder automatically inherits its ability. It processes 30-second audio windows, reducing 50 Hz features by two; windowed processing is not causal sample-by-sample perception. Thinking helps some evaluations and is not uniformly beneficial. Our plan therefore trains audio understanding before optional longer reasoning and tests actual acoustic dependence.

### Music Flamingo

[Music Flamingo v1](https://arxiv.org/html/2511.10289v1), §3 and the training appendices, combines supervision for low-level music properties with richer musical descriptions, timestamp-based embeddings, reasoning warmup and GRPO. Its training extends the language context and uses full fine-tuning for the foundational music model. The useful lesson is that music requires both detailed acoustic cues and relations across time. Long explanations and synthesized annotations are not sufficient evidence of grounded reasoning. This paper is a training/data precedent; its large model and long input budget do not establish our compressed streaming design. Read this arXiv version specifically; an ICLR publication also exists, but was not substituted into the hashed source record.

### MERT

[MERT v5](https://arxiv.org/html/2306.00107v5), §§3–4, combines acoustic-code prediction with Constant-Q Transform reconstruction to preserve pitch-related musical information. It studies 95M and 330M encoders and probes frozen representations on 14 music tasks, including key, pitch, beat tracking and emotion-score regression. That is a useful compact-encoder comparison, not proof of language reasoning or online learning. A 95M music encoder is a credible alternative or offline teacher for our audio frontend. Its benefit must survive token compression; small parameter count alone does not determine streaming cost. Preserve separate sample-rate and temporal-resolution contracts rather than feeding it the speech preprocessing blindly.

### Speech/music emotion transfer has a narrow evidence boundary

[Exploring Acoustic Similarity in Emotional Speech and Music v2](https://arxiv.org/html/2409.17899v2), §§III–V, compares approximately 95M speech and music encoders layer by layer. Its data are acted RAVDESS speech and unaccompanied song, with controlled lexical content. Results vary by layer, domain and emotion category. This supports testing intermediate features and learned layer mixtures. It does not validate universal emotion recognition, instrumental-music interpretation, an individual's private feelings, or subjective feelings in the model. Store perceptual evidence, listener judgments and the model's functional appraisal as different quantities. The earlier [VoxEmo discussion](../sparkx_2_6_product_20261025/Research_Crosscheck.md) remains the companion for uncertainty in listener labels.

## 4. A native spine is feasible, but representation matching must be trained

### Octo: adaptable inputs and passive readouts

[Octo v2](https://arxiv.org/html/2405.12213v2), §III-A–C, uses observation tokenizers, blockwise causal attention and readout tokens. The readouts attend to observations without observations attending back, permitting passive internal readout. Its small diffusion head runs after one backbone pass. New sensor and action specifications are supported through additional encoders/heads and fine-tuning. Retaining pretrained parameter structure does not imply every adaptation experiment froze all pretrained weights. This supports our typed sensory ingress and read-only action/observer egress. It does not establish universal attachment of an untrained adapter or make an internal vector an interpretable thought.

### π0: continuous actions without repeated full language decoding

[π0](https://www.physicalintelligence.company/download/pi0.pdf), architecture and Appendices B/D, adds an approximately 300M action expert to a pretrained vision-language model. It predicts action chunks through flow matching while reusing observation-prefix KV. Its 50 Hz robot operation uses model inference every 0.5 seconds, executing 25 actions between calls; 50 Hz is not the full-model inference rate. This supports separating model decisions, action-head computation and device control clocks. It is neither evidence for safe arbitrary device attachment nor for autonomous lifelong memory. A Spark action interface needs embodiment-specific calibration, training and measured deadlines.

### CALM: native model-to-model exchange needs alignment

[CALM](https://arxiv.org/html/2401.02412v1), §3, trains cross-attention connectors between frozen language models. It supports using hidden representations instead of exchanging only text. Different hidden widths can be projected, but semantic alignment still needs training data. The earlier [pinned implementation review](../sparkx_dynamic_memory_bridge_20261026/Research_Review.md) found that its example shares token IDs between models and identified a masking concern. It is not a ready universal protocol for models with different tokenizers or clocks. Our spine must expose causal feature packets, not assume that another model's hidden coordinates or KV tensors already match Spark's.

## 5. Design consequences and unresolved questions

These are proposed engineering consequences of the combined evidence, not results reported by the papers:

1. Keep intrinsic memory in decoder blocks and the canonical model checkpoint. Give the peripheral spine a separate configuration, state namespace and compatibility test.
2. Train a stable learning program offline, then let persistent native parameters acquire experience during use. Full backbone retraining becomes a versioned maintenance option rather than the only route to persistent learning.
3. Use distinct, tested paths for exact episodic detail and compressed reusable knowledge. More matrices do not eliminate interference, and bounded memory cannot retain an unlimited stream exactly.
4. Start plastic-parameter teaching with small local objectives. Compare end-to-end next-token adaptation only with its entire forward/backward cost charged; frozen backbone weights do not make gradient propagation free.
5. Train visual and audio evidence into the same event/identity system while retaining modality-specific details. Transfer and correspondence need counterfactual tests, not just caption accuracy.
6. Treat representation alignment, causal availability, update ownership and real-time scheduling as part of the spine contract. A matching tensor shape establishes only structural compatibility.

The repository's latest inspected reports still reject useful learned memory and vision quality. The [new repository observation](../../../reports/sparkx_intrinsic_learning_spine_20261026/repository_snapshot.json) found the relevant run artifacts unchanged from the previous review, while preserving the newer repository HEAD. No literature result overrides those local failures.

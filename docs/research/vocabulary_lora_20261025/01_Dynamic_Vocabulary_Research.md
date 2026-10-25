# Dynamic vocabulary research for native learning

Spark should support a growing set of concepts without requiring a growing softmax for every observation. A new name, a new concept, a shorter encoding and a new reasoning skill are different learning problems. The economical design gives each an appropriate representation.

## Four meanings of vocabulary adaptation

| Mechanism | What changes | Suitable use | Main cost or limitation |
|---|---|---|---|
| Lexical memory | A native memory record binds an existing token sequence to a concept | Names, identifiers, new definitions, object identities | Addressing and reliable readout must be learned |
| Vocabulary extension | New permanent token IDs and embedding/output rows | Frequent, stable terms that currently fragment badly | Training both reading and generation; distribution and checkpoint compatibility |
| Dynamic tokenization | Segmentation and embeddings depend on the observed input | Compressing known prompts and documents | Merge selection, embedding synthesis, positions, cache identity and causal training |
| Byte or character hierarchy | Local encoding/decoding surrounds a more sparsely executed global backbone | Broad orthographic robustness and adaptive compute | Substantial architectural conversion and training |

An internal concept slot need not be a text token. For example, a visual object can have an identity and temporal trajectory before anyone names it. Later, “veltraxon” can bind to that identity while its spelling still uses existing text tokens. Learning the word's meaning should not wait for token promotion.

The [swappable-vocabulary follow-up](05_Swappable_Dynamic_Vocabulary.md) adds a temporary input/output view of these lexical entries. Its reusable physical slots are separate from permanent token IDs. It also examines trained atomic phrase generation, complementing the input-only compression research below.

## Research findings

### Dynamic tokenization and tokenizer transfer

**Retrofitting Large Language Models with Dynamic Tokenization**, ACL 2025, adapts segmentation to an input batch and synthesizes embeddings with a hypernetwork. The paper separates prefix compression from generation: retaining the original output embeddings worked best in its prefix experiments. Its large-vocabulary generation experiment lost quality, including 5.9 percentage points on English MMLU. The direct implication for Spark is to investigate input-only compression first. This is not evidence that arbitrary new output tokens work without training. [Paper v3](https://arxiv.org/html/2411.18553v3), [ACL publication](https://aclanthology.org/2025.acl-long.1444/).

**Zero-Shot Tokenizer Transfer**, or ZeTT, trains a hypernetwork over sampled tokenizers to predict token embeddings for a frozen language model. “Zero-shot” describes transfer to a new tokenizer after this training. It does not eliminate model-specific preparation. Spark would need its own calibration and training, including a decision about its tied input/output embeddings. A hypernetwork trained for another backbone is not an interchangeable embedding generator. [Paper v2](https://arxiv.org/html/2405.07883v2).

The official dynamic-tokenization implementation repeatedly finds and merges frequent pairs. Its embedding cache is explicitly bounded, allocates full-width embeddings, and asserts CUDA in the inspected implementation. Spark needs separate MPS qualification and accounting for merge latency, cache misses and embedding generation. A sequence-length reduction alone is insufficient evidence of lower end-to-end cost. [Pinned merge implementation](https://github.com/DariusFeher/dynamic-tokenization/blob/6686e72f5daeda09445793a01b710ca58ec9ec47/tokenizations/dynamic_bpe.py), [pinned cache implementation](https://github.com/DariusFeher/dynamic-tokenization/blob/6686e72f5daeda09445793a01b710ca58ec9ec47/tokenizations/hypernet_cache.py).

### Compositional vocabulary reduction

**VocabDiet** represents some inflected forms through a base form and learned transformations. Its retrofit uses staged distillation and a rank-256 LoRA on the last eight layers. The tiny transformation-parameter count does not describe the entire adaptation cost. Reported reductions of roughly 38–45% for some languages refer to the relevant whole-word vocabulary, not the full model vocabulary. Some language results also lose accuracy. For Spark, this is an offline compression branch requiring morphology, tokenizer and tied-weight evaluation; it is not a mechanism for continually overwriting token IDs. [Paper v2](https://arxiv.org/html/2510.17001v2), [Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.1618/).

This suggests an original design opportunity: use composition to initialize a candidate lexical representation, then retain the original spelling and uncertainty. A guessed embedding should remain a hypothesis until the model demonstrates correct use in new contexts. Morphological similarity alone cannot establish semantic equivalence.

### Byte and hierarchical alternatives

**Bolmo / byteification** adds local byte processing around a pretrained global model and uses staged conversion. The inspected preprint uses 9.8B training tokens for its first stage and 39.3B for its second; these are separate exposures. That is a relatively small fraction of pretraining, but a large project for this repository. Spark's tied embeddings also change the conversion economics: removing a separate output table cannot save a second copy that Spark does not have. [Bolmo v2](https://arxiv.org/html/2512.15586v2).

The related **Nature article published 7 October 2026** makes byteification a current research direction worth retaining. Its reported throughput depends on architecture, compression and hardware; those results do not establish an M2 Ultra or Spark speedup. The Nature publication and Bolmo preprint are treated as one research lineage, not independent replications. [Retrofitting language models to operate over bytes](https://www.nature.com/articles/s41586-026-11111-4).

**H-Net** learns hierarchical chunk boundaries and combines local processing with a global model over compressed sequences. Its routing and smoothing mechanisms allow learned segmentation, but require a different computation graph. Recursive sequence hierarchy does not itself demonstrate recursive self-improvement. [Dynamic Chunking for End-to-End Hierarchical Sequence Modeling v2](https://arxiv.org/html/2507.07955v2).

**Byte Latent Transformer**, or BLT, uses entropy-guided patches, a local byte encoder, a global latent transformer and a local decoder. Its entropy predictor and local networks also consume compute; the inspected setup commonly uses a 100M-parameter entropy model. Its experiments motivate byte-level architecture research, not a drop-in tokenizer swap. [Paper v1](https://arxiv.org/html/2412.09871v1).

For this project, byteification is the most directly relevant long-term retrofit route, while H-Net and BLT provide alternative architecture controls. None should delay establishing useful native memory on the current backbone.

## What the Spark tokenizer probe actually found

The text-only probe used the official BF16 release's tokenizer at revision `66e3ae7bee54e0a523d692cb8ef841f35409807b`, with SHA-256 `710cce15cf3565674c499f9413997c6e8101f2bdd96245cff8f0311fb501248c`. It contains 131,072 distinct, contiguous IDs, 91 entries marked special in its added-token metadata and 89 vocabulary strings containing `unused`. These sets differ. Existing reserved strings remain assigned IDs, and chat controls are not all marked special; do not infer permission to reuse them from their names or flags.

| Authored example | UTF-8 bytes | Base tokens | Interpretation |
|---|---:|---:|---|
| `A blue cup is beside the red cup.` | 33 | 9 | Ordinary text is already compact |
| `The object is called a veltraxon.` | 33 | 10 | The novel name occupies four tokens |
| `q_k_v_proj` | 10 | 5 | A frequently reused code identifier could merit compression |
| `000129004287` | 12 | 12 | The tokenizer splits digits individually; numeric changes need arithmetic tests |
| `Héloïse` | 9 | 4 | Byte spelling remains representable |
| `红色杯子在蓝色杯子后面。` | 36 | 7 | This Chinese example is compact already |

All eleven baseline examples round-tripped exactly. The sample is deliberately small and authored for illustration; it cannot estimate language-wide token efficiency or quality.

Adding the literal string ` veltraxon` to an in-memory tokenizer clone assigned ID 131072 and shortened the repeated-name sentence from 20 to 14 tokens. With `single_word=False`, it also matched the beginning of ` veltraxonic`. With `single_word=True`, the leading-space token did not match the same repeated-name sentence. These observations make boundary semantics a required test, rather than a detail to assume from an API flag. A third trial uses the word without the leading space. All trial settings and exact token sequences are retained in the [probe report](../../../reports/vocabulary_lora_research_20261025/tokenizer_probe.json).

No embedding was added to a model. Fewer token IDs in this probe do not establish comprehension, generation ability or wall-time savings. The [reproduction script](../../../reports/vocabulary_lora_research_20261025/probe_tokenizer.py) accepts the hash-checked public tokenizer and performs no model execution or network access.

## Vocabulary policy for a model that keeps learning

Use a stable base tokenizer and a versioned lexical memory initially. Store original UTF-8 text alongside the canonical representation, with explicit normalization rules. Names, source code, leading spaces, composed/decomposed Unicode and numeric strings must not silently collapse into a shared spelling. Several spellings may legitimately refer to one concept; one spelling may refer to several concepts in different contexts.

Promote a term to a permanent token only when it is frequent enough to repay its training and serving costs, has stable boundaries, and retains correct meaning on held-out uses. Begin with 256 rows, with 1,024 as the first proposed cap and 2,048 as a separate budget study. All new IDs start above the original range. When the cap fills, additional concepts remain in lexical memory; deleting a memory entry does not free a permanent token ID for reuse.

The promotion score can be learned from expected future savings and successful use, but structural limits remain deterministic. A candidate's observed frequency, fragmentation, ambiguity, retention cost and output usefulness are features. The model can propose and score promotions; the runtime performs bounded allocation and atomic serialization. That keeps semantic decisions native while preserving an executable resource contract.

For dynamic prefix compression, restrict merge selection to input already available at the prediction boundary. A training scheme that chooses merges using future target text changes the available information. Preserve role delimiters, byte offsets, visual/event boundaries and the mapping between original spans and compressed positions. Cache keys must include the merge plan, tokenizer version, embedding version and relevant memory/adapter state. Compression is a compute hypothesis to evaluate after lexical learning works.

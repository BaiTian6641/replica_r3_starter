# V13 counterfactual diagnosis: research cross-check

Checked online: 30 October 2026. These primary method/experiment sections constrain the next hypothesis; they do not predict Spark accuracy, MPS cost, or the diagnostic outcome.

| Source | Relevant published evidence | Boundary for the v13 successor |
|---|---|---|
| [Wang et al., LongMem, NeurIPS 2023, §§2.1–2.3, 3.1, 3.4](https://arxiv.org/html/2306.07174v1) | LongMem caches attention key/value vectors from a frozen backbone and trains a separate residual SideNet to retrieve and fuse them. Retrieval uses token chunks; the authors report that smaller chunks help fine-grained label retrieval and that useful memory size is task-dependent. The published adaptation uses 26B tokens, a different backbone, layer-18 K/V features and a trained SideNet. | A less-compressed token/chunk-feature control is a reasonable **candidate only if** v13 shows that four-slot compression destroys the relevant distinction. LongMem does not validate four slots, Spark's reader, this 512-update objective, or zero-training token-feature readout. Do not call a token-feature arm LongMem or import its results. |
| [Ge et al., ICAE, ICLR 2024, §§2.1–2.3, 3.1–3.3](https://arxiv.org/html/2307.06945v4) | ICAE trains a LoRA-adapted encoder with autoencoding, language modeling and instruction objectives. Its default is 128 slots. The paper reports worse restoration at higher compression, benefits from pretraining, and initial failure when independently compressed spans were concatenated without multi-span training. | Four-slot v12 compression is a different and much more aggressive hypothesis. If facts vanish before the reader, compare against a bounded less-compressed representation with fresh training and a new budget; do not add LoRA, autoencoding, placement changes and extra slots to the current diagnostic. If facts survive compression, ICAE does not justify changing the representation. |
| [LongMem memory fusion, §2.3](https://arxiv.org/html/2306.07174v1) | Retrieved content is fused through a learned memory-augmented layer with token-specific queries and gating; it is not a generic claim that any attention reader will use a memory bank correctly. | If slots retain counterfactual distinctions but first-token distributions ignore them, isolate query/reader alignment or fusion placement while holding source features fixed. This is a separate mechanism from adding slots. |

## Decision rule

The v13 diagnostic preserves per-pair canonical-source features, compressed slots, pre-projection reader outputs, injected residuals, full first-token distributions and complete answers. It also separates complete-row permutation from the sealed v12 raw slot/address shuffle, semantic wrong-source corruption, zero, absent and exact-text controls.

- **Distinction lost or strongly attenuated by compression:** preregister one bounded less-compressed source-feature arm, with its own training, compute and retention controls.
- **Distinction present in slots but lost at reader/residual/logits:** preregister one reader/query-alignment or fusion-placement arm with the source representation fixed.
- **Distinction reaches logits but complete answers fail:** keep the representation and reader fixed; inspect response termination/scoring rather than enlarging memory.

The sealed v12 shuffle gate and terminal 24/32 result are unchanged. No development labels, density promotion, persistent-learning claim, or second simultaneous mechanism change is allowed. Exact-evidence retrieval remains a charged reference, not evidence of parametric learning.

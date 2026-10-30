# V12 contextual compression: primary-research cross-check

Checked online: 29 October 2026. Scope: primary method and experiment sections, not abstracts alone. Author-reported results below were not reproduced on Spark and do not establish useful v12 recall, MPS cost, or a diagnosis of v11's fitting failure.

| Primary source | What the published mechanism actually does | Boundary for the v12 candidate |
|---|---|---|
| [Ge et al., ICAE v4, §§2–3](https://arxiv.org/html/2307.06945v4) | A LoRA-adapted LLM encoder appends learned memory tokens and produces memory slots for an untouched decoder. Training uses autoencoding plus language-model pretraining and then instruction fine-tuning. The default configuration uses 128 slots, not four pooled slots. §3.3.3 reports that concatenating independently compressed spans initially failed until multi-span examples were included in training. | Supports investigating learned source compression with a frozen decoder; does not support assuming that frozen final hidden states plus a small compressor will be sufficient. V12 changes the encoder, slot count, objectives, placement, and training scale. Two-span ordering/masking and fresh-world response quality are therefore required evidence, not implementation details to assume away. |
| [Chevalier et al., AutoCompressors v2, §§3–4](https://arxiv.org/html/2305.14788v2) | Special summary-token output states become soft prompts. Summary vectors accumulate across segments and are trained with next-token loss; later segments stop gradients after a declared truncation. Published models fine-tune the LM or use LoRA and large token budgets. Longer plain context can outperform compressed vectors. | Output hidden states are relevant only as part of a trained compression system. The paper does not validate stateless region pooling, four slots per arbitrary span, or v12's teacher objective. Its explicit truncation and caching costs reinforce the need to declare v12's detach boundary and charge source recomputation rather than treating cached features as free. |
| [Mu et al., Gist Tokens v3, §§2–6](https://arxiv.org/html/2304.08467v3) | Gist tokens are inserted into the input and attention masks force prompt information through their activations. The language model is fine-tuned on many tasks; cached gist prefixes are then reused. Reported FLOP/storage reductions do not always yield comparable wall-time gains. | This is a different mechanism from canonical-source-only frozen features and post-block memory readers. It motivates matched wrong/shuffled/absent controls and actual total-cost measurements, but supplies no expected v12 accuracy or latency benefit. A smaller slot count is not evidence of lower total cost after adding a frozen source pass. |

## Decision

Retain frozen Spark source features followed by live trainable contextual compression as a **bounded hypothesis**. Do not add autoencoding, LoRA, special memory tokens, gist masking, recursive accumulation, or extra training objectives to the current v12 experiment: each would be a separately versioned mechanism with a new protocol and budget.

The v12 acceptance path must independently establish:

1. canonical-source-only feature extraction and exact disabled-extension parity;
2. live response-loss gradients through compression and every active reader after zero-initialized outputs open;
3. ordered two-span/eight-slot behavior and correct/wrong/shuffled/zero/absent controls;
4. fresh fitting-group performance before any development label is opened;
5. source-pass, cache-miss/restart, tokenizer, retrieval, training, generation, storage, and latency costs;
6. no claim that related paper results transfer to Spark, four-slot values, the declared two-site reader, or this machine.

The prior [Spark memory research cross-check](Research_Crosscheck.md) remains authoritative for formation, uTTT/MemT, nested-learning, and evaluation boundaries. This supplement adds compression-specific evidence; it neither rewrites historical results nor authorizes a native launch.

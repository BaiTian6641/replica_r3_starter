# Spark pre-MLP fusion placement v14

Status: **native attempt01 is running under one exclusive MPS worker.** No fitting, development, restart, utility or promotion result is available yet.

## Predeclared mechanism

The only primary mechanism change is intrinsic-memory injection at the MLP input (`block_input_query_to_pre_mlp_residual`) instead of after the complete decoder block. Block-input queries, sites 3/7, frozen canonical-source features, four ordered width-256 slots per span, reader parameters, NativeFusedAdamW, initialization seed 17, learning rate, objective, data order and 512-update schedule are unchanged from v12 attempt10. The extension starts from that same initialization, not from the trained attempt10 weights.

The sealed post-block v12 outcome remains the control: contextual 24/32 with development closed. This arm must not be compared by selecting a better prefix.

## Fixed gates and exposure

1. Frozen complete-context calibration must pass 8/8 with zero false recall and no truncation.
2. Frozen fitting teacher must pass 32/32 before detached top-64 targets are admitted.
3. Exactly 512 updates run in the declared order. Only the final probe may open development; every eight-world fitting group must independently pass. No best-prefix selection.
4. Fitting freezes and teacher targets evict before development. Both development groups and every compound family must independently satisfy the unchanged utility criteria.
5. Correction and revoked-read behavior remain separate from cache invalidation and learned unlearning.
6. Only extension plus optimizer state is saved. A fresh process must restore exact logits and perform one continuation update with original parameter hashes unchanged.

Development labels remain unread unless final fitting passes. Persistent-P updates, density arms, capacity expansion and promotion are not authorized.

## Exact bindings and envelope

- Protocol: `configs/spark_pre_mlp_v14.json` (SHA-256 `944b87baa0400c1b016d61aac3a996cf1c45f3a3d235791f371b3d288ccb6272`).
- Packet: `reports/intrinsic_memory_pre_mlp_v14/native_packet_two.json` (SHA-256 `074bee9bdc0b8f77f892ac491b3465e4fcb56c022bf7a5cca17af381751af7ff`).
- Implementation: SHA-256 `21159bc05c74554cc9f07597abeb537f2af5d61904dfb13e173127a5c9969b70`.
- Authorization: `reports/intrinsic_memory_pre_mlp_v14/authorization_two.json`; one attempt, zero retries.
- Source: sealed official BF16 with the established tokenizer and original parameter identities.
- Budget: 23 stages; 7,155 reserved stage seconds inside 7,200 outer seconds; 9,778 forwards; 949 source-feature forwards; 513 backwards/updates; 7,256 generated tokens; 4,669 prompt tokenizations; 3,618 writes; 55,490,098,857 reserved read bytes; 205,537,280 artifact bytes.
- Resources: one exclusive worker, 50 GiB sampled driver, 512 MiB sustained swap growth, at least 15% host free, CPU fallback disabled.

## Evidence to append

Terminal raw answers, every fitting group, counterfactual pairs, completion/truncation, live/durable parity, original-weight checks, restart, development exposure, token/tool/action counts, latency, memory, artifacts, unsuccessful trajectories and independently reconciled outcomes will be appended after attempt01 terminates. CPU pre-MLP mechanics pass 96 focused tests, but those tests establish only null parity, gradients, placement-specific influence and checkpoint identity—not native learning.

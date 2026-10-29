# Spark memory: mechanism cross-check, 29 October 2026

This is a current supplement to the preserved 26 October research review.
Primary HTML methods were checked for the four core mechanisms below; the
additional evaluation papers were checked at abstract scope. Author results
are not reproduced Spark evidence. The [S0–S1 protocol](../../experiments/20261029_Spark_Memory_S0_S1.md)
is a Spark-specific experiment informed by these mechanisms.

| Source and inspected scope | Mechanism and boundary | Project consequence |
|---|---|---|
| [Titans v1, §§3.2–3.3](https://arxiv.org/html/2501.00663v1) | Adaptive associative neural memory includes momentum/forgetting; “persistent memory” separately includes input-independent learned task tokens. | Do not equate those tokens with a durable personal experience store. Our linear delta control is not a Titans reproduction. |
| [uTTT v1, §2 and appendix C.2](https://arxiv.org/html/2610.05484v1) | Layers read chunk-start state. Boundary writes have sequential and aggregated variants; fast weights reset to learned initialization between sequences. Later losses train earlier differentiable writes. | Our proposed aggregate-once schedule is one declared choice. Shared storage alone, cross-session persistence or an imperative detached update does not reproduce the trained mechanism. |
| [MemT-1.0 v2, §§2.3 and 4.2–4.3](https://arxiv.org/html/2610.26825v2) | Generation reads the incoming snapshot. Storage incorporates the completed input/response for the next interaction. Mixture-of-Storage is dense across experts. | Token-level storage is not continual read/write inside the same generated reasoning sequence. Within-reasoning formation and sparse top-k are separate project hypotheses. |
| [Nested Learning / HOPE v1, §§3.2–3.3 and 9.1](https://arxiv.org/html/2512.24695v1) | Levels have context flows, objectives, update rules and knowledge transfer. The pretrained Llama adaptation includes 15B continued-pretraining tokens. | Timers and storage tiers alone are insufficient. The paper does not establish that an untrained Spark retrofit is useful or affordable here. |

[Titan-Llama v1](https://arxiv.org/abs/2610.19210v1) studies post-training a memory
retrofit of attention-based models. Its abstract supports the feasibility of
investigating retrofits; it provides no measured Spark result or latency on this
machine. Our original attention, positional behavior, gated attention, GELU and
tied output path remain preserved.

The proposed functional-control tests distinguish prompting from trained
correction. [Huang et al., v2](https://arxiv.org/abs/2310.01798v2) report that
unaided revision can degrade reasoning; [SCoRe v2](https://arxiv.org/abs/2409.12917v2)
reports gains after an explicit multi-turn RL procedure. These motivate measuring
both beneficial and harmful revisions, not assuming fluent reconsideration helps.
[Kadavath et al., v4](https://arxiv.org/abs/2207.05221v4) report useful confidence
signals but calibration difficulties on new tasks. Capability self-knowledge must
be calibrated independently; it is not approved identity.

[LongMemEval v2](https://arxiv.org/abs/2410.10813v2) covers multi-session and
temporal reasoning, updates and abstention. [LoCoMo](https://arxiv.org/abs/2402.17753v1)
provides long conversational histories grounded in personas and event graphs.
[TRACE](https://arxiv.org/abs/2310.06762v1) studies continual learning and forgetting.
They are candidates for separately admitted external evaluation, not default
training corpora, and none alone identifies the internal memory mechanism or RSI.

All dimensions, sites, ports, update counts, clock periods and acceptance margins
in our unrun stages are project choices. Measured engineering results are kept
in the [dated evidence ledger](../../../reports/intrinsic_memory_v3/evidence_ledger_20261029.json).
No paper comparison changes those results or authorizes a native run.

# Prompt-fixed augmented generation — 22 October 2026

**Passed:** native uncached D1/D2 generation compatibility on one actual 256-token fixture. The [completed qualification report](../../reports/minicpm5_augmented_generation_qualification256_report.json) records exact greedy token and next-token-logit parity with the unwrapped sealed MiniCPM5 source, one-time context construction, independent branch sensitivity, and nonzero sidecar export/reload parity. This implements the generation portion of the [declared D1 protocol](20261022_D1_Fixed_Context_Protocol.md); no optimizer update or quality improvement is established by this check.

The FP32 MPS run used source revision `f97400052a43d642bbc6e9975e2397e3ae6a6b52`, with one 2,516,756,480-parameter backbone resident at a time. Fixture `capacity-256-dev-world-0` has 256 full training tokens, comprising a 254-token native prompt and two reference answer/EOS tokens; its observed support contains 197 tokens. Generation receives only the original prompt and observed support, never the reference answer. The tested sidecars contain 895,490 parameters in 48 tensors, with 64-wide memory/workspace, four slots, two workspace steps and final-layer insertion.

| Check | Observed result |
|---|---|
| D1 null branch, originals frozen | Eight generated tokens exactly match native; all 1,044,480 next-token logits match with maximum absolute difference `0.0` |
| D2 null branch, originals eligible | Same exact eight-token and 1,044,480-logit parity |
| Forced multi-token path, each arm | EOS stopping disabled for three steps; exact tokens and all 391,680 next-token logits match |
| Decision-context lifecycle | Source writer and workspace encoder each called once per request; input IDs/masks unchanged; module modes restored and context released |
| Memory-only nonzero projection probe | First-token logits change finitely; maximum absolute difference `0.0773773` |
| Workspace-only nonzero projection probe | First-token logits change finitely; maximum absolute difference `0.0620061` |
| Both projections nonzero, export then reload | Eight generated tokens and 1,044,480 next-token logits match exactly, maximum difference `0.0` |

The independent projection probes use deterministic `sin(arange) × 0.02` weights. They demonstrate that both branches can affect logits; these are diagnostic perturbations, not trained sidecars. The nonzero export probe retained the same eight greedy tokens as the native reference. Observable logit sensitivity does not require an argmax change and does not establish better answers.

Both native and null-augmented outputs were `To solve this, we need to determine`. All eight-token normal-policy checks reached their declared cap without EOS. They are matching **partial continuations**, not successful complete integer answers; the fixture's reference answer is `2`. The separate forced-three-step check validates multi-token context reuse independently of EOS behavior. No task accuracy claim follows from these outputs.

The complete profiled process took **24.180 seconds**, including source validation, reference generation, both arms, probes, export, backbone release and reload. Sampled peak live MPS allocation was **9.516 GiB**, driver allocation **10.145 GiB**, process-lifetime peak RSS **14.597 GiB**, and sampled peak RSS **8.034 GiB**. Swap occupancy was 1,562.69 MiB at both endpoints; the reported host-free percentage changed from 89% to 86%. Samples were taken at 0.25-second cadence and reported no errors. These overlapping unified-memory counters must not be added; sampling can miss transients, and unchanged swap endpoints do not measure intervening traffic. [Original memory samples](../../reports/minicpm5_augmented_generation_qualification256_memory_samples.jsonl).

The [qualifier](../../scripts/qualify_augmented_generation.py) captures native references before installing wrappers and explicitly uses `logits_to_keep=0` so both paths compute the same full-prefix output projection shape. The [generation API](../../src/replica_lab/generation.py) recomputes the growing causal prefix with `use_cache=False`, while source slots and the original-prompt workspace remain fixed for the request. This is a correctness reference, not KV-cache optimization or a concurrent serving interface.

The scoped next gate is native D1 optimizer learning and generated tiny overfit, followed separately by D2 training capacity and fair architecture controls. This compatibility run does not establish useful memory, retention, frozen-base self-improvement, broad transfer or Replica fidelity.

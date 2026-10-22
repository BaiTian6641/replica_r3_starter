# S4/S5 — separate native VLM, optional memory and later hardware

Reviewed **22 October 2026, America/Chicago**. See the [master roadmap](00_Master_Roadmap.md), [current execution state](../../execution_state.md) and [dated primary-source review](07_Research_Evidence_20261022.md). Gates and budgets are proposed unless a linked existing report establishes execution. Planned artifacts below do not claim files already exist.

The small V4.7 VLM is a separate native hybrid text/vision port. Its processor-expanded image tokens, visual insertion, positional logic, recurrent state and attention cache must be preserved. A Llama text wrapper, image filename in memory text or caption-only score is not a VLM training result. Distinct full_text/full_vlm parameter inventories make training claims reviewable.

Optional Titans/Hope/uTTT/Engram/latent interfaces remain one-at-a-time hypotheses. The functional linear fast-state primitive is an initial comparison, not a reproduction. Every depth port reads one chunk-start snapshot; completed writes publish after the chunk, without storage mutation during activation recomputation. State owner/reset and producer compatibility are explicit.

The RTX PRO 4000 Blackwell SFF is a later 24 GB target, not installed hardware here. The current full-text FP32 mixed-state budget alone is 26.40 GB decimal before activations/temporaries. CUDA dtype support, mixed Muon partition/offload behavior, host RAM/staging and restart require qualification. Ordinary DDP or inference device_map does not solve that budget. Larger model ports inherit source records and tested protocols; their latent interfaces, calibrators, baselines and memory encodings must be trained/measured anew.

## S4.1 — Pin and qualify the native small VLM

**Status:** `pending`. **Dependencies:** S2.5, S1.1.

Resolve/seal MiniCPM-V-4.7-1B model, processor and native code revisions. Inventory text/vision/projector groups; load actual image inputs and test native forward/generation plus hybrid recurrent/attention cache behavior before adding memory.

- **Artifact:** planned: V4.7 source/processor lock and tensor inventory; planned: native image-dependent parity/cache baseline.
- **Interface:** Image asset manifest -> causal asset loader -> native processor expansion/mRoPE/visual insertion -> hybrid model; separate from LlamaMemoryAdapter Llama integration.
- **Test gate:** Changing relevant pixels changes the expected answer, and cache/re-prefill results agree within tolerance. Processor-expanded lengths, image crops/tokens and recurrent/attention states are traced and restart-owned.
- **Resources:** One small VLM resident; first CPU/tiny contract then bounded native Mac profile. Pin new environment separately if native requirements conflict with the text lock.
- **Failure policy:** Stop at unsupported native kernels/caches and document exact limitation. Caption text alone never qualifies as multimodal execution.
- **Next smallest task:** Pin the native V4.7 revision and inspect processor output shapes for one synthetic image/question pair.

## S4.2 — Implement distinct full_text and full_vlm training scopes

**Status:** `pending`. **Dependencies:** S4.1, S0.4.

Build a V4.7-specific adapter/collator preserving native hybrid states. Explicitly register decoder/embeddings/head, vision, projector and new modules. Test full_text with declared frozen vision separately from full_vlm with all original modal parameters eligible.

- **Artifact:** planned: multimodal collator and parameter registry; planned: image-gradient/update and full_vlm restart audits.
- **Interface:** Causal text/image episode -> native collator/loss masks -> modal group optimizer -> native save/processor reload. No image-reference string stands in for a tensor.
- **Test gate:** Actual vision/projector gradients and changed tensors in full_vlm; full_text reports frozen groups honestly. Response masks, native caches, source causality, export/reload and resume pass.
- **Resources:** Microbatch 1, measured text/image budgets and checkpointing; reduce image count/resolution only as declared protocol variables.
- **Failure policy:** Do not silently freeze modal groups to fit. Preserve a full_text result separately; add residency changes for any failed full_vlm claim.
- **Next smallest task:** Trace an image-dependent loss backward to named vision and projector parameters on the smallest native batch.

## S4.3 — Evaluate multimodal memory and chronology controls

**Status:** `pending`. **Dependencies:** S4.2, S1.4.

Release a separate proposed 3,000-target tranche with eligible multimodal personas and solver-backed visual chronology. Compare native/evidence-only/augmented systems with true images, text-only, shuffled images and removed images.

- **Artifact:** planned: modal episode and asset release manifest; planned: image-dependence, temporal and cost ablation report.
- **Interface:** Asset lineage + episode cutoff -> modal collator -> fixed multimodal evaluator; image/task families inherit source split.
- **Test gate:** Useful gain depends on relevant pixels and chronological evidence, survives image/shuffle controls, and includes modal parameter/compute costs. No caption-only or future-frame leakage claim.
- **Resources:** Separate text/image token budget from the 20k text pilot. Start a small solver-backed visual slice before optional public assets.
- **Failure policy:** Fix asset availability/cutoff/processor bugs or record no visual benefit. Do not inflate text evidence by silently adding image-derived gold captions.
- **Next smallest task:** Generate paired visual scenes differing in exactly one decisive attribute with independent expected answers.

## S4.4 — Test one optional memory mechanism at a time

**Status:** `pending`. **Dependencies:** S3.5.

Choose one hypothesis from functional linear fast state, Titans-style update, uTTT shared ownership, Engram lookup, Hope multi-rate updates or frozen latent interface. Declare state owner, loss, reset, chunk barrier and matched private/capacity/compute controls.

- **Artifact:** planned: single-mechanism preregistration and version-pinned source notes; planned: causal/state-isolation/backend and transfer ablation report.
- **Interface:** Observed chunk-start snapshot -> all depth reads -> combined post-chunk writes -> session/branch-owned state; published model parameters remain independently versioned.
- **Test gate:** No future-token influence, cross-session contamination, storage writes during checkpoint recomputation or stale source encodings. Mechanism adds useful held-out benefit beyond the simplest control at stated memory/compute cost.
- **Resources:** Begin linear-state CPU/tiny MPS tests; one optional mechanism per run. Do not import CUDA/Triton or higher-order requirements into the stable text env without qualification.
- **Failure policy:** If no extra value, omit the mechanism. Do not activate all research papers simultaneously or block the learned loop awaiting a reproduction.
- **Next smallest task:** Write a falsifiable one-mechanism hypothesis and an adversarial chunk-boundary causality case.

## S5.1 — Qualify Muon8bit on the actual Blackwell host

**Status:** `waiting_for_target_hardware`. **Dependencies:** S0.4.

Inventory actual RTX PRO 4000 Blackwell SFF UUID/memory, host RAM, driver/CUDA/PyTorch build and native dtype support. Reproduce tiny CPU/CUDA optimizer routing, INT8 state, checkpoint and D0/D1/D2 updates before native training.

- **Artifact:** planned: Blackwell hardware/environment lock; planned: CUDA tiny optimizer, dtype and resume parity report.
- **Interface:** Same optimizer equations/named eligibility and source lock -> CUDA backend; any BF16 autocast or residency change is separately declared.
- **Test gate:** No CPU fallback, all intended groups updated, state/resume correct and independent numerical tests pass. Inventory confirms actual available memory; vendor 24 GB is not a measured free-memory value.
- **Resources:** Later target only; no installed CUDA device assumed on this Mac. Use a separate platform lock/environment and one bounded tiny job.
- **Failure policy:** Resolve build/kernel incompatibility before full model. Keep Mac evidence valid and do not infer CUDA success from it.
- **Next smallest task:** On arrival of the workstation, capture GPU UUID, nvidia-smi and PyTorch device/dtype probes.

## S5.2 — Implement a full-scope 24 GB residency solution

**Status:** `pending`. **Dependencies:** S5.1, S0.5.

Implement training-aware CPU optimizer offload or qualified ZeRO/FSDP partitioning for the mixed Muon/AdamW registry, or separately validate a lower-residency layout. The current FP32 full-text persistent 26.40 GB budget already exceeds card capacity before activations.

- **Artifact:** planned: offload/partition placement and dtype manifest; planned: full D0/D2 CUDA memory/transfer/scope/resume report.
- **Interface:** Named full registry -> sharding/offload adapter -> native wrapper/optimizer -> checkpoint portability. Ordinary DDP and inference device_map do not pool training memory.
- **Test gate:** All originals including embeddings/head stay trainable in D0/D2, actual changed-tensor audit passes, finite updates and resume meet tolerance, and measured peak/host staging/transfers fit the declared capacity.
- **Resources:** 24 GB target GPU plus inventoried host RAM; measure pinned buffers/PCIe transfers and CPU copies. First 256 real tokens then 512. Muon partition semantics require explicit validation.
- **Failure policy:** If full scope cannot fit, preserve the failure and redesign residency; label modules-only/LoRA alternatives honestly. Never freeze layers while calling it full fine-tuning.
- **Next smallest task:** Build a per-group weight/gradient/state/temporary placement ledger and test offload on the tiny mixed optimizer.

## S5.3 — Port only earned mechanisms to larger frozen models

**Status:** `pending`. **Dependencies:** S2.5, S3.5, S5.2.

Select one larger target from the declared Gemma/MiniCPM-V roster after reconfirming official architecture/license/revision. Build new readers/calibrators and native cache/processor adapters; re-encode approved source memories using the new producer.

- **Artifact:** planned: per-model source/environment/compatibility manifests; planned: native baseline and frozen-interface/candidate profile.
- **Interface:** Portable source records + protocol + verified skill semantics -> new model-specific encoders/interfaces; small-model latent tensors are not transferred as aligned memories.
- **Test gate:** Native baseline and null parity, exact frozen-base hashes, compatible candidate publication, measured fit/latency and held-out gains beyond evidence-only controls. Larger native capability may require harder fresh tasks.
- **Resources:** Profile storage for all MoE parameters, not active-per-token count. Default larger base frozen; qualify actual inference/extension-training residency before campaign.
- **Failure policy:** Stop a model port if memory or interface quality fails; do not silently change target spelling, use active parameter count as storage, or claim transfer from source-record reuse alone.
- **Next smallest task:** Choose one earned small-model mechanism and define the corresponding larger-model native insertion/ownership contract.

## S5.4 — Run longer operation with migration and audit

**Status:** `pending`. **Dependencies:** S5.3, R4.

Extend chronological operation with versioned source/branch histories, held-out skill transfer, drift, source correction/deletion, rollback and model migration. Replenish promotion windows independently and reserve fresh audit families for final reporting.

- **Artifact:** planned: long-horizon retention/fidelity and cost matrix; planned: model/source migration, rollback and rebuild playbook.
- **Interface:** Accepted manifests + source lineage + branch journals -> reproducible deployments -> independent chronological evaluator and human-source fidelity review.
- **Test gate:** Useful later behavior and retained critical tasks at declared costs, traceable branch divergence, successful rollback/rebuild and no hidden audit reuse. Report negative/inconclusive periods and changing source preferences.
- **Resources:** Extend horizon from ten cycles only after reliable instrumentation; determine cap from measured wall time and evaluator independence, not arbitrary perpetual running.
- **Failure policy:** Pause publication on hard invariant failure while retaining the last accepted manifest. Restore independent evaluation before restarting candidate acceptance.
- **Next smallest task:** Review ten-cycle failure modes and define which longer-horizon uncertainty the next bounded campaign will resolve.

# SparkX 2.6 4B final-product consolidation

**27 October 2026 · Architecture version 5 · Research and documentation only.**

Read the [unified final-product blueprint](../../docs/architecture/23_SparkX_2_6_4B_Final_Product_Blueprint.md) for the full specification. It provides one coherent entry point for architecture, native learning, system design, training, data, validation and deployment. The [declarative JSON](../../configs/sparkx_2_6_4b_final_product_v5.json) records the reference profile without changing runtime admission.

## Short assessment

The proposed product is technically plausible as a research and engineering program. The strongest foundation is full offline multimodal training followed by explicitly bounded native experience/parameter updates. The evidence does **not** establish that the combined model will learn reliably, meet its intended cost targets or fit native Spark training on this host.

The reference keeps Spark's original 4.112B backbone and adds compact visual/audio frontends, nine intrinsic memory sites, episodic records, fast associative state, sixteen persistent plastic experts with two clocks, a recurrent functional core and a swappable phrase overlay. The optional neural spine is separate from those native memory components.

MiniCPM remains the gate before an expensive Spark campaign. Current memory quality, acceptance calibration and visual alignment results are insufficient. The independent continuation test proves only exact state restore for a 16-expert plastic bank; it does not demonstrate a useful learner. No result in this review admits local full Spark training.

## Current evidence at a glance

| Evidence | Current result | What it does not establish |
|---|---|---|
| Memory acceptance v4 | `shared_mean/causal_query_refresh` is **rejected**: 21/384 accepted-correct facts and precision 21/138 over 138 accepted. Coverage and fail-closed checks are valid; the separate oracle recovers 384/384 answerable facts. | Useful learned memory quality, writer training or promotion. [v3 remains historical](../../reports/memory_acceptance_20261026/causal_fresh_restart02_acceptance_v3.json) and is never accepted as v4. |
| Acceptance calibration | Development-only calibration selects **`oracle_cannot_meet_gates`** and stops. No value point passes all gates; held-out rows are unscored. | A calibrated confidence policy, held-out confirmation, native execution or writer training. |
| Independent plastic continuation | Three fresh processes produce **`status=exact_match`** for a **16-expert, 28-event** plastic bank: 57 A/A states, exact boundary and final canonical states, zero guard stops and zero swap growth. | Production 381-tensor MiniCPM continuation, a numeric envelope, learning quality, retention or promotion. |
| Vision learnability curve | The completed curve returns **`curve_no_optimization_progress`**. Answer/caption NLL fall by 65.40%/69.68% (T1), but both rise at prefix 1 (T2 fails). Answer correct and prefill-gold accuracy stay **0/14**; caption F1 gain is 0.028. | Development data, held-out evaluation, V0, promotion or retry. The trailing 4→7 descent is descriptive only, and the prefix-7 endpoint is not a decision input. |
| Final verification | The latest [final verification and clean-clone receipt](../../reports/final_verification_20261027/summary.json) records a clean checkout at `bece2a8f` with zero failures and 48 explicit artifact-dependent skips. Its workspace run retains four historical vision migration-pin failures rather than relabelling them as green. | Native model behavior, training admission or capability; the receipt predates this reconciliation and is not a claim that these new edits were cloned. |

Sources: [acceptance v4](../../reports/memory_acceptance_20261026/causal_fresh_restart02_acceptance_v4.json), [v4 receipt manifest](../../reports/memory_acceptance_20261026/receipt_manifest_v4.restart02.json), [historical acceptance v3](../../reports/memory_acceptance_20261026/causal_fresh_restart02_acceptance_v3.json), [calibration rejection](../../reports/memory_acceptance_20261026/calibration_dev01.json), [continuation comparison](../../reports/nested_text_continuation_comparison_20261026/comparison.json), [vision curve summary](../../reports/vision_learnability_curve_v1_20261026/summary.json), [final verification and clean clone](../../reports/final_verification_20261027/summary.json), [historical clean export](../../reports/clean_export_20261026_final/summary.json).

## What this consolidation resolves

| Prior ambiguity | Version 5 decision |
|---|---|
| Repeated architecture addenda | A single 26-section product blueprint with short sections, tables and a whole-model diagram |
| Multimodal acquisition versus self-learning | Full offline training establishes perception/reasoning; deployed learning uses those abilities |
| Memory as a peripheral | `E/F/P`, readers and learning rules are intrinsic; UAP remains a separate extension interface |
| “Dynamic parameters” without an algorithm | Explicit `F` recurrence, selected full-rank `P` updates, replay objectives, first-order outer estimator and clocks |
| More ports without a causal schedule | Nine-site reference; two-pass correctness baseline; alternatives require quality/cost qualification |
| Audio treated as instantaneous streaming | Pinned SPEAR configuration is noncausal; complete-window availability is explicit |
| Emotion, truth and personality conflated | Observed acoustics, perceived affect, appraisal and approved personality are separate |
| Swappable IDs corrupting memory | Stable concept/lexeme identity; recyclable slots with generations; atomic mapping and cache rules |
| Retention confused with parameter learning | Separate `E`, `F`, `P` and restart interventions; joint `M×P×V` validation |
| Approximate budgets treated as a final count | Reconciled partial tensor ledger, explicit omissions and separate training/serving accounts |

## Fresh source verification

The review retrieved **19 public source artifacts** and rechecked the hashes of **10 retained research artifacts**. The [source manifest](evidence.json) records all 29 bindings, retrieval times, URLs, hashes and the relevant methods examined. Source text remains in the research cache, not copied into the repository.

Required source checks now run from **tracked repository pins**, not the temporary cache. The pinned Spark configuration and modeling code are committed under `reports/spark_preparation_20261025/run_01/assets/official_bf16_control/`; their hashes equal the source-manifest bindings. The SPEAR `causal:false` extract is bound to the same recorded source hash. Cache files are optional cross-checks only, so a clean checkout never depends on `/private/tmp`.

Key cross-checks:

- The pinned Spark implementation uses a **GELU-gated original MLP**, per-head sigmoid attention gating, 4096-wide attention concatenation and a tied embedding/head. The added plastic branch may use a different activation without changing the original branch.
- SPEAR v2 supplies a compact general-audio/speech path, but `causal:false` means a closed-window delay. Its English-heavy pretraining does not certify multilingual listening.
- MemT supports the case for intrinsic memory integration. Its specific attention geometry, DeltaNet update and storage mixture do not validate our width-512 residual reader or sparse routing.
- HOPE supplies the multilevel-learning motivation. The proposed initial first-order plastic learner is deliberately identified by its actual approximation.
- Explicit visual grounding has independent 4B-model evidence. DeepSeek's inspected report uses a much larger MoE and its own compression stack; its original provenance remains unresolved. The mirrored report is supporting material, not the sole foundation.
- DVAGen supports dynamic phrase representations. Stable lexical identity across memory, optimizer state and slot turnover remains a new integration problem that this plan tests directly.

These are source/method checks, not reproductions of paper results.

## Parameter and memory reconciliation

The [reproducible calculator](calculate_costs.py) reads the v5 JSON and the prior inventory, not prose. Run:

```sh
python3 reports/sparkx_final_product_20261026/calculate_costs.py
python3 reports/sparkx_final_product_20261026/verify.py --root . --out reports/sparkx_final_product_20261026
```

It removes older two-reader alternatives and the legacy query/output term, substitutes nine compact Q/K/V/O readers, adds plastic down/up projections, and counts both phrase-encoder projections. The partial inventory is **4,347,805,444 stable parameters + 12,582,912 plastic parameters = 4,360,388,356**. Additional unsealed modules remain; 4.4B is a rough planning placeholder.

The live/staging dynamic-state calculation still matches the v4.5 resident-Adam account: about **159.60 MiB live / 190.39 MiB illustrated peak** with optional address adaptation, before the separate replay allowance and omitted backend/activation/payload costs.

The KV account distinguishes **342 MiB** at an 8192-position working context with bounded sliding caches from **1.125 GiB** when every layer retains the full context. A local attention mask does not prove bounded allocation.

For full training, a 4.4B FP32-weight/gradient/two-moment Adam payload alone is about **65.57 GiB**. The current local 50 GiB process guard remains incompatible with that full Spark ownership layout; preparation and precision research do not change that admission. BF16/Muon/distributed forecasts remain conditional research choices, not measured fit and not changed training defaults.

## Evidence, drift and bundle discipline

The [input snapshot](repository_snapshot.json) hashes **40** reviewed inputs and captures HEAD/worktree status. The [portable evidence pins](portable_evidence.json) list every required current receipt, configuration and source pin with repository-relative paths. The [reviewed-change record](reviewed_input_changes.json) separates the five explicitly reviewed architecture/pointer/receipt changes from any unexplained drift. Unexplained snapshot or historical-manifest drift now **fails current qualification**; reviewed changes remain separate records rather than being silently absorbed.

The [117-input first-tranche manifest](../../reports/sparkx_first_tranche_20261026/research_bundle_manifest.json) is the reviewed current bundle: the SparkX v5.1 residency receipt refresh regenerated it from current bytes, and its prior pre-migration bytes are preserved separately as `reports/sparkx_first_tranche_20261026/research_bundle_manifest_pre_sparkx_v5_1_residency.json`. Current evidence is pinned by the additive `reports/sparkx_final_product_20261026/final_product_bundle_manifest.json`, generated as:

```sh
python3 scripts/prepare_sparkx_research_bundle.py --profile final-product --write
python3 scripts/prepare_sparkx_research_bundle.py --profile final-product --check
```

That profile is a hashed manifest, not an asset copy. It includes the current v4, calibration, continuation, vision-curve, final-verification and clean-export receipts, plus explicit clean-export links. It never includes a temporary source-cache path and never rewrites the historical profile's 117-input set; the four re-reviewed inputs are the blueprint, this guide, the verifier and the bundle generator. The v5.1 residency receipt is a reviewed current update in the first-tranche manifest lineage, whose pre-migration bytes are preserved separately.

The [20-item requirements map](requirements_traceability.json) connects every major requested feature to its design sections, validation and current evidence status. The blueprint's open-issue table names the experiments needed to close unresolved choices rather than presenting them as settled capabilities.

## Scope and verification

This review changes documentation and declarative research artifacts only. It does not start training, acquire model weights or dataset media, import native model code, modify the protected runtime file, stage/commit changes, or alter resource guards and experiment admission.

The portable verifier writes [verification_v2.json](verification_v2.json) and preserves the historical [verification.json](verification.json). It checks document links/anchors, JSON consistency, tracked source pins, pinned configuration/code agreement, inventory arithmetic, current evidence denominators and verdicts, reviewed drift, the first-tranche manifest, and the additive final-product manifest. Every required check is self-contained in tracked evidence; missing source-cache entries are reported as unavailable rather than described as verified.

These receipts do not certify model quality, runtime fit, learning or native Spark training. The final product entry point is version 5. [Document 22](../../docs/architecture/22_State_Mechanism_and_RSI_Contracts.md) continues to own state/transaction/RSI authority, and actual experiment protocols continue to govern execution.

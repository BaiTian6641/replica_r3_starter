# SparkX 2.6 4B final-product consolidation

**26 October 2026 · Architecture version 5.0 · Research and documentation only.**

Read the [unified final-product blueprint](../../docs/architecture/23_SparkX_2_6_4B_Final_Product_Blueprint.md) for the full specification. It now provides one coherent entry point for architecture, native learning, system design, training, data, validation and deployment. The [declarative JSON](../../configs/sparkx_2_6_4b_final_product_v5.json) records the reference profile without changing runtime admission.

## Short assessment

The proposed product is technically plausible as a research and engineering program. The strongest foundation is full offline multimodal training followed by explicitly bounded native experience/parameter updates. The evidence does **not** yet establish that the combined model will learn reliably or meet its intended cost targets.

The reference keeps Spark's original 4.112B backbone and adds compact visual/audio frontends, nine intrinsic memory sites, episodic records, fast associative state, sixteen persistent plastic experts with two clocks, a recurrent functional core and a swappable phrase overlay. The optional neural spine is separate from those native memory components.

MiniCPM remains the gate before an expensive Spark campaign. Current memory quality and visual alignment results are insufficient. NL0's state/math tests help make the implementation trustworthy but do not demonstrate a useful learner.

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

The review retrieved **19 public source artifacts**: pinned model code/configurations/cards and relevant paper/report versions. It also rechecked the hashes of **10 retained research artifacts**. The [source manifest](evidence.json) records all 29 bindings, retrieval times, source URLs, hashes and the relevant methods examined. Source text remains in the research cache, not copied into the repository.

Key cross-checks:

- The pinned Spark implementation uses a **GELU-gated original MLP**, per-head sigmoid attention gating, 4096-wide attention concatenation and a tied embedding/head. The added plastic branch may use a different activation without changing the original branch.
- SPEAR v2 supplies a compact general-audio/speech path, but `causal:false` means a closed-window delay. Its English-heavy pretraining does not certify multilingual listening.
- MemT supports the case for intrinsic memory integration. Its specific attention geometry, DeltaNet update and storage mixture do not validate our width-512 residual reader or sparse routing.
- HOPE supplies the multilevel-learning motivation. The proposed initial first-order plastic learner is deliberately identified by its actual approximation.
- Explicit visual grounding has independent 4B-model evidence. DeepSeek's inspected report uses a much larger MoE and its own compression stack; its original provenance remains unresolved. The mirrored report is supporting material, not the sole foundation.
- DVAGen supports dynamic phrase representations. Stable lexical identity across memory, optimizer state and slot turnover remains a new integration problem that this plan tests directly.

These are source/method checks, not reproductions of paper results. Technical claims in the blueprint link to primary papers and author code/cards; the DeepSeek mirror is explicitly labelled.

## Parameter and memory reconciliation

The [reproducible calculator](calculate_costs.py) reads the v5 JSON and the prior inventory, not prose. Run:

```sh
python3 reports/sparkx_final_product_20261026/calculate_costs.py
python3 reports/sparkx_final_product_20261026/verify.py
```

It removes the older two-reader alternatives and legacy query/output term, substitutes nine compact Q/K/V/O readers, adds plastic down/up projections, and counts **both** phrase-encoder projections. The resulting partial inventory is **4,347,805,444 stable parameters + 12,582,912 plastic parameters = 4,360,388,356**. Additional unsealed modules remain; 4.4B is a rough planning placeholder.

This differs from the earlier approximate 4.3630B projection because the architectures/formulas differ. Historical receipts are preserved. The live/staging dynamic-state calculation still matches the v4.5 resident-Adam account: about **159.60 MiB live / 190.39 MiB illustrated peak** with optional address adaptation, before the separate replay allowance and omitted backend/activation/payload costs.

The new KV account distinguishes **342 MiB** at an 8192-position working context with genuinely bounded sliding caches from **1.125 GiB** when every layer retains the full context. Actual backend behavior must be measured. A local attention mask does not prove bounded allocation.

For full training, a 4.4B FP32-weight/gradient/two-moment Adam payload alone is about **65.57 GiB**. The current local 50 GiB process guard therefore remains incompatible with that full Spark ownership layout. New BF16/Muon/distributed forecasts remain conditional research choices, not changed training defaults.

## Repository evidence and status discipline

The [input snapshot](repository_snapshot.json) hashes **40** reviewed repository inputs and captures HEAD/worktree status. Architecture v4.5, the state contract and index are preserved as pre-edit text snapshots. The snapshot predates the v5 pointer edits; `verification.json` records those changes under `input_changes_since_snapshot`.

The latest reviewed acceptance v3 rejects native memory quality, while its underlying runtime report independently says completed/inconclusive with restart parity and zero updates. The blueprint preserves both statements. The vision projector diagnostic remains unpromoted. NL0 has a software/math qualification only. No combined audiovisual nested learner is claimed to have passed.

The [20-item requirements map](requirements_traceability.json) connects every major requested feature to its design sections, validation and current evidence status. The blueprint's open-issue table gives the experiments needed to close unresolved choices rather than presenting them as settled capabilities.

## Scope and verification

This review changes documentation and declarative research artifacts only. It does not start training, acquire model weights or dataset media, import native model code, modify the protected runtime file, stage/commit changes, or alter resource guards and experiment admission.

[Verification results](verification.json) record **223 passed checks** covering document links/anchors, JSON consistency, source hashes, pinned configuration/code agreement, inventory reconciliation and arithmetic. They do not certify model quality, runtime fit or learning. Re-running source-hash checks requires the cached source artifacts; missing cache entries are reported as unavailable rather than described as verified.

The final product entry point is version 5. [Document 22](../../docs/architecture/22_State_Mechanism_and_RSI_Contracts.md) continues to own state/transaction/RSI authority, and actual experiment protocols continue to govern execution.

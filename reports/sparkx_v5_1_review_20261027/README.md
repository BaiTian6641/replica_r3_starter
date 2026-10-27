# SparkX v5.1 and MiniCPM5 direction review

**27 October 2026 · Documentation/research deliverable. No native run or training.**

The supplied refinement draft was reviewed as a proposal. Its embedded instructions to a workspace agent were not treated as authorization to execute training, change optimizer policy or rewrite earlier experiment receipts.

## Read first

- [Updated unified Spark blueprint](../../docs/architecture/23_SparkX_2_6_4B_Final_Product_Blueprint.md).
- [Reviewed v5.1 changes and expanded native learning design](../../docs/architecture/24_SparkX_2_6_v5_1_Refinement_Plan.md).
- [MiniCPM direction-correction and validation program](../../docs/architecture/25_MiniCPM5_Direction_Correction_and_Learning_Capacity_Plan.md).
- [Non-executable design configuration](../../configs/sparkx_2_6_v5_1_direction_design.json).

## Main conclusions

The useful direction corrections are richer visual evidence, progressive full multimodal training, separate retrieval/readout diagnosis, native experience storage and bounded deferred consolidation. The new capacity proposal is P32 on MiniCPM, P64 on Spark, and P128 as a later study. Sparse active execution keeps expert matrix work bounded; optimizer state, bandwidth and learning passes remain material costs.

An additional **Mkv** branch tests a 65,536×256 learned value table with product-key addressing. It is intrinsic parametric memory, distinct from exact E records and disposable KV. Compare it independently against experts before stacking both. The new plan also specifies learned update-strength experiments, optional nonlinear F, future-query outer objectives and a later separate learning-policy improvement loop.

The draft's strongest claims needed correction:

- Persistent Adam momentum is not established as a bug. The cited reset paper studies deep RL; reset variants need separate bias-correction and restart semantics.
- SDFT's theoretical reverse KL and practical forward-KL experiments must be distinguished. The revised paper and authors' pinned code identify the practical direction.
- Self-generated teaching and entailment judgments do not independently verify factual truth.
- Dense expert reads, sparse writes and null-space projection are separate experimental factors. A nonlinear gate/up edit does not inherit AlphaEdit's guarantee.
- The pinned NaFlex processor defaults to 256 patches. A 1024-patch path requires an explicit override and geometry checks.
- Four new address keys must replace the old key in E accounting. Entity anchors, hot packets, optimizer state and staging cannot be omitted.

All sixteen C1–C16 items have a disposition in [claim_dispositions.json](claim_dispositions.json). The attachment bibliography contains **109 detected labelled entries**; [its inventory](attachment_reference_inventory.json) distinguishes selected source checks from entries not revalidated here. The draft's claim that all titles were checked is not inherited as our verification.

## Current evidence and direction

The exact selected-record fixture passed 148/152 held-out rows with host lookup and zero updates. Native learned-memory acceptance remains rejected. The P response diagnostic remains 0/28 on free-generation train fit despite lower NLL. Independent small-bank continuation passed. The vision reproduction supports first-step overshoot and remains 0/14 on generated answers.

Consequently, the Mini plan repairs learnability/readout first, tests capacity and nested objectives second, and combines already-trained vision/audio with memory later. It preserves old NL/V identifiers and adds DC study identifiers. No past failure was relabelled as success, and no old held-out data was reopened for this review.

## Capacity account

| Proposed profile | P weights / two moments | E2 payload | Partial live subtotal | Proposed dynamic envelope / separate replay |
|---|---:|---:|---:|---|
| Mini P32 | 96 / 192 MiB | 24.5 MiB | 312.5 MiB | 512 / 128 MiB |
| Spark P64 | 192 / 384 MiB | 130 MiB | 706 MiB | 1024 / 256 MiB |
| Spark P128 | 384 / 768 MiB | 130 MiB | 1282 MiB | 2048 / 512 MiB |

Optional Mkv-64k adds 64 MiB values plus 128 MiB moments. A dense-gradient fallback adds another 64 MiB. Selected two-expert update staging is 24 MiB; selected 256-row Mkv staging is 1 MiB only with a genuinely row-sparse implementation. Mini's combined P32+Mkv arm has a separate proposed 768 MiB dynamic envelope. These are partial analytical accounts, not measured fit or revised process guards.

The reconciled v5.1 residency inventory is [Doc24 §4.4](../../docs/architecture/24_SparkX_2_6_v5_1_Refinement_Plan.md#44-residency-inventory-measured-analytical-and-unsealed). It labels retained vision tower, base/extended weights, optimizer state, activations, restart transients, P16/P32/P64 and Mkv-4k/64k as `measured_native`, `analytical_estimate`, `unsealed` or `not_measured`; every row keeps `full_training_qualified = false`. The official BF16 release (7,843.17 MiB of BF16 storage), official FP8 storage (native execution refused on MPS), the dequantized BF16-storage `derivative_02` (no per-token F8 quantization) and native FP8 execution (never run) are distinct identities, not interchangeable evidence. [native_gates.json](native_gates.json) lists the unresolved native measurements separately.

Later Mkv capacity arms reach 67.11M and 268.44M value parameters at 256k/1M rows, with 768/3072 MiB of FP32 values and Adam moments. They require separate profiles and migration qualification; top8 reads do not make residency or lookup bandwidth free.

[analytical_capacity.json](analytical_capacity.json) includes E2 schema bytes, profile counts, reader KV, expert MACs, dense retrieval-query work, optional nonlinear F and media-position examples. Recompute with:

```bash
python3 reports/sparkx_v5_1_review_20261027/calculate_capacity.py
```

The old 4.4B full-training estimate is explicitly historical. So400m's public aggregate includes a text tower; this review does not invent a sealed vision-only count or final v5.1 total. Hardware provisioning must be repeated for the actual graph.

## Source and document lineage

[evidence.json](evidence.json) records **23 primary records**: eighteen freshly fetched records and five retained primary sources rechecked for architectural connections. These comprise sixteen paper records and seven model/configuration/code records. Review depth is stated per source; this is not a claim of independently reproducing their experiments or reading every appendix.

The source cache contains paper bodies, model metadata and author code only. No model weights, datasets or media were downloaded. Source hashes are retained here; cached full paper text is not required by the portable documentation verifier. Web citations link directly to primary pages or pinned author code.

[repository_snapshot.json](repository_snapshot.json) pins the initial HEAD, the attached draft and eleven prior/current repository inputs. Their exact bytes are preserved under [inputs](inputs/). These snapshots are historical source material, not current execution instructions. The current authored outputs are separately pinned in [deliverables.json](deliverables.json).

[migration_manifest.json](migration_manifest.json) makes the lineage explicit. Each changed existing document has an old/new identity pair, hashes and a reason; each added document, configuration and review record is a declared new identity; unchanged historical controls and identifiers are retained. The same manifest pins the unversioned `runs/` receipts cited by chapter 21 as local artifacts that a clean checkout is not required to contain. No validator was weakened and no historical receipt was overwritten.

Updated existing documents: the main Spark blueprint, MiniCPM chapter 18, state-contract evidence/pending-namespace note, chapter 21's entry link, the architecture index and the earlier nested-learning guide. The v5 executable/reference configuration, historical reports and experiment receipts were left unchanged. Other tasks' working-tree files were preserved.

## Verification scope

The portable verifier is deterministic and clean-checkout aware: it reads repository-relative paths only, permits the generated `verification.json` and the declared local-only run receipts to be absent, and fails on unexplained untracked references. It checks source metadata completeness, preserved input hashes, local document links and anchors, configuration/report references, all sixteen dispositions, current output hashes, capacity/residency arithmetic and migration old/new identity integrity. It also confirms the historical v5 configuration and retained identifiers are unchanged, and that the migration digest reproduces. These are documentation/arithmetic checks; they do not qualify learning, perception, sparse kernels or model fit. Native gates are loaded separately from [native_gates.json](native_gates.json) and can only be reported as unresolved here.

```bash
python3 reports/sparkx_v5_1_review_20261027/verify.py --root . --tracked auto
python3 -m pytest tests/test_sparkx_v5_1_review.py -q
```

`--tracked auto` accepts a declared new identity before its first commit and requires tracked status afterwards; use `--tracked require` on a committed clean checkout. Results are written to `verification.json` and a second run over the same tree is byte-identical. The tamper tests mutate document bytes, config arithmetic, migration hashes, a preserved input, a local link and a native-gate status, and require each mutation to fail. Native model execution, optimizer tests, full-suite tests, dataset intake, checkpoint migration and training were not run because this change is a research/design update.

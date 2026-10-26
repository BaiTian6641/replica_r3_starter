# SparkX nested-learning audit report

**26 October 2026 · Research/documentation review. No model execution, training, dataset admission or capability promotion.**

## Finding

The existing architecture had a credible collection of components but an incomplete specification of how they learn together. Revision 4.5 supplies a concrete candidate, resolves conflicts with the repository optimizer contract, and adds experiments that distinguish retrieval, persistent parameter learning, learned adaptation and policy improvement.

The architecture remains a 4B original language backbone plus additional native perception/memory/learning parameters. Full offline fine-tuning establishes multimodal competence. Deployed learning updates declared intrinsic state/parameters while stable release weights remain frozen. Optional sensory/model exchange through the spine remains separate.

## Research coverage and evidence quality

The review inspected **16 primary papers** and **two official optimizer references**. Nine paper full texts were newly fetched in this review; seven were reused from earlier hashed research captures and revisited for the applicable mechanism. The [manifest](evidence.json) records individual URLs, versions, retrieval timestamps, hashes and sections. Full third-party papers are not copied into Git; they live in a declared, gitignored content-addressed source cache.

The `scratch_path` and `text_path` fields in the manifest are **historical provenance only**. Verification never reads payload bytes from them, and provisioning does so only when a caller explicitly supplies that root. Payload resolution uses the cache declared in [the sources config](../../configs/sparkx_nested_learning_sources_v1.json), keyed by the evidence SHA-256.

The strongest connections are HOPE's explicit optimization levels and transfer edges; TTT/TTT-E2E and OML's training for later adaptation; MemT's intrinsic memory integration; and uTTT's shared-depth state and write schedule. MIRAS helps specify the objective/retention/update choices. ATLAS motivates a bounded recent-context comparison, but its particular Muon algorithm conflicts with the admitted optimizer policy and is not silently substituted.

Newer multimodal research adds two useful failure tests: evidence-use forgetting despite stable answer accuracy, and misleading transfer results when domains are easy to route or only one task order is used. AutoMMemo was explicitly excluded as proof of native plastic learning because its searched memo programs operate outside the backbone.

These are mechanisms and experimental cautions from different settings. No source demonstrates this complete Spark visual/audio/vocabulary/plastic-memory system at the target resource cost. The detailed [comparison](../../docs/research/sparkx_nested_learning_20261026/Research_Review.md) distinguishes publication findings from our proposed combination.

Two retrieval issues were resolved rather than hidden: La-MAML has v2, not the initially attempted v3; the stable AdamW documentation URL returned only a redirect, so the explicit documentation version was fetched. No conclusions rely on those failed/redirect-only responses.

## Source cache and verification modes

Payloads are content-addressed under the ignored default cache root `data/cache/sparkx_nested_learning_20261026`:

```text
sources/<evidence-sha256><source-suffix>   # raw HTML/PDF/pinned source bytes
text/<text-key>.txt                       # extracted text
```

`<text-key>` is a declared `text_sha256` when the manifest supplies one, otherwise the source SHA-256. The config declares 18 required sources, disabled network access, the cache layout, and explicit `unprovisioned` / `mismatched` statuses. It is ignored by Git, so a clean checkout contains no third-party payloads.

Verify metadata and committed arithmetic without any payloads:

```bash
python3 reports/sparkx_nested_learning_20261026/verify.py --metadata-only
```

This mode inspects no cache file and always reports `source_payload_verified: false` and `source_payload_status: "unprovisioned"`. It never labels a source verified. Document/config/link/cost results are reported separately from the payload result.

Verify strictly, requiring all 18 payloads:

```bash
python3 reports/sparkx_nested_learning_20261026/verify.py
```

Strict mode fails clearly with the missing or mismatched source names. It checks names, sizes, hashes, declared extracted-text hashes and cache confinement. There is no fallback to the absolute historical paths.

Override the cache root with `--cache-root PATH` or `SPARKX_NESTED_LEARNING_SOURCE_CACHE=/path`. Write the full JSON receipt explicitly; the committed [verification.json](verification.json) is the historical document/config/link/arithmetic receipt and is not silently replaced:

```bash
python3 reports/sparkx_nested_learning_20261026/verify.py \
  --output reports/sparkx_nested_learning_20261026/verification.json
```

Provision offline from a user-declared local directory, or from the historical scratch root only when it is named explicitly:

```bash
python3 scripts/provision_sparkx_nested_sources.py \
  --from-dir /path/to/declared/source-directory

python3 scripts/provision_sparkx_nested_sources.py \
  --historical-scratch-root /private/tmp/spark_av_experience_20261026
```

The provisioner never downloads. It accepts flat files named by the historical basename, by `<sha256><suffix>`, or by source name, and also accepts a `sources/`+`text/` cache layout. It verifies every input, copies to a staging area, re-verifies the staged bytes, and only then publishes. A clean cache root is published by one atomic directory rename; an existing cache uses atomic no-overwrite per-file publication. Matching files are skipped; a mismatched existing file aborts rather than being replaced. It declares a 64 MiB total and 16 MiB per-file write budget and uses no `/private/tmp` path unless one is supplied. See [provision_sparkx_nested_sources.py](../../scripts/provision_sparkx_nested_sources.py).

The self-contained tests use synthetic payloads and temporary caches; they require no third-party text, model, network or native job.

## Repository reconciliation

The [input snapshot](repository_snapshot.json) records HEAD `e05b0217808cec5f838672519bd368ec3beea0d5`, working-tree status and ten input hashes. Concurrent project work was present and retained. The [revision 4.4 snapshot](architecture_v4_4_snapshot.txt) and [state-contract snapshot](state_contract_snapshot.txt) preserve the pre-edit versions; earlier research receipts were not rewritten as new passes.

The newest reviewed vision diagnostic performs 24 projector-only updates, with 3/14 train-fit and 2/18 development answers and zero caption-following accuracy. It stops before audit. Earlier memory quality remains rejected at 110/240 recall, while fresh-process restart parity is an engineering pass. This review does not infer product learning from either result.

Dataset metadata validation and six synthetic intake fixtures are recorded separately from real-media admission. The first tranche's 1,798 bytes are protocol fixtures, not evidence of an acquired training corpus. The current dataset catalog remains a source-selection artifact.

The protected runtime implementation was not opened, imported or changed. No runtime implementation, training scripts, model weights, execution registry or existing experiment result was changed by this review.

## Major corrections

| Earlier ambiguity | Revision 4.5 decision |
|---|---|
| More components/update rates treated as enough | Each learning level now has context, objective, clock, reset, persistence and transfer edges. |
| `P` loss and consolidation schedule underspecified | Behavioral reference; explicit local-feature candidate; 8+8 plastic experts with a sixteen-teaching-event slow clock and matched controls. |
| Earlier SGD proposal versus current optimizer policy | All new actual parameter training follows `NativeFusedAdamW`; explicit functional `F` recurrence is a separate mechanism. |
| Exact meta-learning assumed available | PyTorch 2.14.1 fused Adam rejects differentiable steps. First-order `P` credit is declared; exact short-unroll training needs separate qualification. |
| Optimizer state omitted from small profile | Resident moments and atomic candidate state are charged. A larger research envelope replaces the misleading 128 MiB fit implication. |
| Cached local loss treated as useful learning | Require source-free behavioral transfer and cost/quality comparison to decoder-backward learning. |
| Several memory owners could write per layer | Immutable shared reads and one aggregated observation transition; explicit historical-gradient test. |
| Vocabulary swap could change knowledge identity | Concept/lexeme identity remains stable; n-gram keys use canonical base tokens, not recycled slots. |
| Correct answers treated as grounded retention | Repeat valid spatial, temporal, acoustic and AV interventions after learning. |
| Correction implied exact forgetting | Checkpoint/replay fallback and explicit limitations for compressed state. |
| Continued learning implied recursive improvement | Separate fresh-family tests for learning-policy revisions after fixed-policy success. |

## Cost audit

[calculate_costs.py](calculate_costs.py) uses only Python standard-library arithmetic. Its [output](analytical_costs.json) is a reproducible payload lower bound, not a device measurement.

| Item | MiniCPM candidate | Spark candidate |
|---|---:|---:|
| Plastic parameters | 12,582,912 | 12,582,912 |
| Plastic FP32 weights | 48 MiB | 48 MiB |
| Plastic Adam moments | 96 MiB | 96 MiB |
| Live state, including optional adaptive address weights/moments | 146.54 MiB | 159.60 MiB |
| Illustrated atomic-update peak | 170.89 MiB | 190.39 MiB |
| Illustrated peak plus 64 MiB replay | 234.89 MiB | 254.39 MiB |

The new candidate envelope is 256 MiB mutable state plus 64 MiB replay, with optional raw evidence separately bounded. Backend temporaries, activations, KV, metadata overhead, additional snapshots and feature-extraction buffers are not covered by the table's lower bound. Dense gradients over all experts would also exceed the selected-expert estimate. Actual peak and latency qualification remain required; no host guard changed.

For initial full fine-tuning, the separate [residency v2 ledger](../sparkx_residency_20261026/residency_v2.json) correctly distinguishes 16-byte all-FP32 training from the **18-byte** BF16-storage plus FP32-master/gradient/two-moment configuration. Its declared integrated inventory requires about 64.385 GiB and 72.433 GiB respectively, before activations. These exceed the current 50 GiB process guard. A larger qualified training environment may therefore be needed even if the deployed learner is inexpensive. FP8 storage is not proof of a usable full-training backend.

The calculator also contains a scalar, one-step AdamW differentiation illustration. It confirms that the analytic derivative matches finite differences while the first-order approximation differs. It uses deliberately illustrative scalar settings and no ML library; it is **not** an optimizer implementation test or a model-learning result.

## Deliverables and remaining work

- [Reading guide](../../docs/research/sparkx_nested_learning_20261026/README.md)
- [Research comparison](../../docs/research/sparkx_nested_learning_20261026/Research_Review.md)
- [Detailed learning contract](../../docs/research/sparkx_nested_learning_20261026/Nested_Learning_Contract.md)
- [Validation ladder and 22-item issue ledger](../../docs/research/sparkx_nested_learning_20261026/Validation_and_Open_Issues.md)
- [Machine-readable design candidate](../../configs/sparkx_2_6_4b_nested_learning_v1.json)
- [Source cache declaration](../../configs/sparkx_nested_learning_sources_v1.json)
- [Offline source provisioner](../../scripts/provision_sparkx_nested_sources.py)
- [Updated consolidated plan](../../docs/architecture/21_SparkX_2_6_4B_Consolidated_Architecture_and_MiniCPM_Validation.md)

The issue ledger explicitly preserves the major open obligations: useful perception and memory, admitted real-media episodes, source-free parameter transfer, outer-training benefit, schedule advantage, efficient local updates, native resource fit, correction limits and later policy improvement. The next implementation step is the contract/gradient checks followed by a small controlled text-learning study, alongside repair of the existing capability failures.

## Non-claims

The [verification receipt](verification.json) reports document/config/link/source-integrity/arithmetic checks only. Source hashes prove byte identity to the recorded capture; they do not prove that a paper's claims are correct or reproduced. Strict success does not mean any proposed neural mechanism ran or passed a capability gate. Metadata-only success is weaker by design: payloads were not inspected, and no source is labelled verified.

This review does not download papers at verification time, does not commit third-party full text, and does not execute a model, optimizer, training run or native job. It must not be cited as evidence of product learning, recursive improvement or resource fit.

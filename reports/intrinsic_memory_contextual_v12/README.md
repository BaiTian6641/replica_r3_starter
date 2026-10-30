# Contextual intrinsic memory v12 — CPU implementation report

Status: implementation and entry points delivered by the v12 implementer.
**No behavioural test, CPU workflow run, native model load or native attempt was
executed by the implementer.** The coordinator runs the verification below after
integration. Nothing in this directory is native evidence or a capability claim.

Coordinator history: 61 tests passed before independent review and 69 after the
V1–V5 repairs. After cycle 4, **79 checks passed**, including the two cold-failure
reproductions. Fresh `run03` passes all 2/9/36 layouts with live gradients,
distinct controls, true live-versus-release and two-record cold value/logit
parity, publication failure recovery and exact fresh-process continuation.
Measured parent wall was 16.135751 s and peak RSS 769,294,336 bytes; run03 has
244 files / 102,306,331 bytes. `native_admissible` remains `false`: this is
synthetic CPU engineering evidence, pending final independent re-review.

## Scope delivered

| Area | File |
|---|---|
| Frozen canonical-source provider, bounded C cache, live compressor, source work accounting | `src/replica_lab/contextual_source_features_v12.py` |
| Distinct v12 identity, provenance, authority/correction contract | `src/replica_lab/contextual_feature_identity_v12.py` |
| Producer/order/reader owner and complete extension checkpoint | `src/replica_lab/contextual_memory_layout_v12.py` |
| Durable reads, cold tickets, publication and cost wrapping | `src/replica_lab/contextual_memory_v12.py` |
| Immutable producer-release corpus | `src/replica_lab/contextual_release_v12.py` |
| Tiny random 36-block BF16 synthetic fixture and tokenizer | `src/replica_lab/contextual_v12_fixture.py` |
| CPU workflow (2/9/36 reader arms, fresh-process continuation) | `scripts/run_intrinsic_contextual_cpu_v12.py` |
| Tests and shared test helpers | `tests/test_contextual_source_features_v12.py`, `tests/test_contextual_memory_layout_v12.py`, `tests/test_contextual_memory_v12.py`, `tests/test_contextual_v12_integration.py`, `tests/v12_fixtures.py` |

Sealed v3–v11 modules, receipts and `src/replica_lab/runtime_pilot.py` were not
read, imported, edited or staged. No shared module was modified.

## Exact intended verification commands

```bash
.venv/bin/python -m pytest -q \
  tests/test_contextual_source_features_v12.py \
  tests/test_contextual_memory_layout_v12.py \
  tests/test_contextual_memory_v12.py \
  tests/test_contextual_v12_integration.py

# one fresh output directory; refuses to overwrite
.venv/bin/python scripts/run_intrinsic_contextual_cpu_v12.py \
  --output reports/intrinsic_memory_contextual_v12/run01
```

Machine-readable delivery hashes, execution status and limitations are in
`implementation_status.json` in this directory.

The workflow uses only the CPU, `torch.set_num_threads(1)`, no download and no
pretrained checkpoint. It writes per-arm `run.log`, `child.log`, `result.json`,
`restart.json`, `expected.json`, pending descriptors, stores, release generations
and a complete extension checkpoint, plus `result.json` and an artifact
inventory at the output root. The workflow does **not** run pytest; it records
the command above in its own result.

## Synthetic status

- Backbone: tiny random Spark-shaped BF16 fixture, all 36 blocks available,
  explicit causal attention, GELU MLP and tied output table.
- Tokenizer: `SyntheticTokenizer` (whitespace/alphanumeric, immutable
  `identity()`; special tokens and truncation are refused).
- Sources, questions, answers, corrections and revocations are synthetic.
- Teacher labels are not used. No v11 development label is opened.
- The gradient probes are step 1 (expected zero upstream gradients from the
  zero-initialized reader output projections) and step 2 (live producer, order,
  ordered slots and every active reader). The permutation ("shuffled") control
  is measured after eight updates because a pure slot permutation is
  order-invariant under near-uniform attention and can round to identical BF16
  residual logits at step 2; the workflow records both the slot difference and
  the measured logits. The fresh-process continuation is update 9.
- Engineering influence on random weights is not semantic recall, useful
  readout, persistent-P learning, promotion, RSI or native readiness.
- `spec.original_parameters_sha256` in CPU fixtures is a declared fixture
  digest. A native packet must verify the full original bytes separately and
  attest that content identity to the provider; without attestation the provider
  performs one bounded content hash at construction.

## Ownership and detach diagram

```
frozen Spark source (base, eval, no_grad)
  └─ adapter.forward_hidden(canonical span tokens only;
                           use_projector=False, use_ports=False,
                           use_spine=False, intrinsic_memory=None)
       └─ Intentional detach boundary (provider.features -> FP32 frozen states)
            └─ LIVE trainable compressor (producer owner):
               LayerNorm -> Linear(hidden,256) -> position -> 1 encoder layer
               -> four ordered softmax token-region pools -> width-256 slots
                 └─ order embedding (owner: order)
                      └─ ONE immutable LayerMemorySnapshot per forward
                         (cloned, masks padded slots; no retrieval/write inside)
                           └─ one SpinePort per active site (owner: reader;
                              block-input query, post-block residual)
                                └─ frozen original decoder continuation
                                   (no no_grad; original params frozen)
```

- `NativeFusedAdamW` owns `producer`, `order` and `reader` exactly once each.
- Original Spark parameters have zero optimizer ownership and are never
  serialized; `contains_base_weights` is `false`.
- Durable reads/admission/checkpoint export detach. Live training tensors stay
  connected from the compressor through ordered slots to every active reader.
- The 32 MiB / 64-entry source-feature cache is C: bounded, in RAM, never
  serialized, never authority, and cleared on correction, revocation detection,
  publication, reload, `close()` and explicit `clear_source_caches()`.

## Source work accounting interface (native operator)

`replica_lab.contextual_source_features_v12`:

- `SourceWorkRequest(operation, source_bytes, max_tokens, tokens)`
  with operations `source_tokenize` and `source_forward`.
- `SourceWorkAccounting` protocol: `reserve(request) -> token` before the work,
  `complete(token, status='completed'|'failed', actual=..., detail=...)` after.
- `source_work_amounts(request)` — the declared counter mapping.
- `WorkLedgerSourceAccounting(ledger, stage)` — maps onto the durable
  `intrinsic_memory_budget_v5.WorkLedger`: one tokenizer pass reserves
  `source_encodes=1` and `input_tokens=source_bytes` (upper bound) before
  tokenization; a cache-missing forward reserves `forwards=1` and its exact
  `input_tokens` before `forward_hidden`.
- `FrozenSourceFeatures(adapter, tokenizer, spec, *, accounting)` requires the
  provider. There is no in-process counter-only fallback: a missing or rejecting
  accounting provider fails before dispatch. Completion never refunds the
  reservation (uncertain/restarted work remains charged).
- `adapter_content_identity(adapter, *, content_identity=None)` and
  `adapter_contract_digest(adapter, *, content_identity=None)` bind the provider
  to the exact frozen adapter content, not only geometry or mutation stamps. A
  native operator that already verified the source artifact should attach
  `adapter.v12_adapter_content_sha256` (or pass `content_identity`) so the
  provider does not hash the backbone again. Without an attestation the
  tiny/custom backbone is hashed once, tensor by tensor, with bounded temporary
  storage. `spec.source_execution_sha256` must equal the content-bound contract;
  `tokenizer_contract_digest(tokenizer)` must equal `spec.tokenizer_sha256`.
- `SourceFeatureSpec.identity()` is the durable feature contract. Operational
  cache bounds are excluded from it. The value provenance carries
  `feature_contract_sha256`, the declared `original_parameters_sha256`, the
  token sequence digest/count and ordered pool regions under the store's
  1024-byte lineage bound (the full adapter/tokenizer contracts are
  transitively bound by the feature-contract digest and appear explicitly in
  the checkpoint manifest).
- `encoder.release()` is the complete **representation release**: producer,
  order and reader state plus the feature contract. Any optimizer update to the
  contextual path invalidates durable values. The checkpoint records
  `producer_release`/`representation_release` (both the full release used by
  batches and the release corpus) and `producer_state_release` (the
  compressor-only digest). The outer checkpoint manifest also pins
  `contains_base_weights=false`, `original_weights_serialized=false`,
  `fitting_frozen`, the feature/adapter/tokenizer/layout contracts and
  `components_manifest_sha256`; load requires exact equality before mutation.
- Failed warm/cold reads evict only the records whose own authority or payload
  is invalid. `ContextualRecordError` identifies a single stale-release/authority
  failure so a healthy sibling selected in the same batch keeps its bounded C
  cache. Cold staging reads each record with accounting and returns per-record
  invalid IDs; failed `resume` performs no secondary uncharged payload probe and
  persists the typed `ContextualReadFailure` even if best-effort eviction raises.
  `evict_record(full_span_record_id)` evicts exactly that record and returns the
  number of cache entries removed; a record-group prefix is not accepted.
- Source authority includes named non-parameter buffers as well as parameters.
  A buffer-only or parameter mutation clears C, and `save()` revalidates the
  frozen original and ownership before creating the export directory or writing
  any bytes.
- Token and feature cache keys include the actual slice `text_digest`; two spans
  with identical geometry/hash metadata but different text never share IDs or
  features. `canonical_sha256` remains a whole-record digest (not a per-slice
  digest), so a synthetic mismatched span is tokenized by its real text.
- Release manifests bind `source_identity` to `path.source_identity` at `read`,
  `values` and `validate_batch` boundaries. `values(..., source_identity=)` and
  `validate_batch(batch, scope=..., source_identity=)` require the keyword.
  A failed immutable release write removes partial bytes and releases its
  artifact charge before re-raising; a retry publishes a new generation.
- Durable slot order sorts semantically by `(source_id, source_epoch,
  source_order)`; `record_id` is only the deterministic fallback for true ties.
  Reversed caller order, post-correction, two-record live/warm/cold/release
  parity, pointer/manifest publication failure and whole-release eviction are
  tested.
- `active_sites` accepts any iterable; empty tuple/list/generator skips memory
  work and matches the base exactly, and nonempty iterables are normalized to
  tuples before the reader sees them.
- CPU workflow evidence names the comparisons accurately:
  `release_reads_consistent` is release-to-release; `live_vs_release_exact` and
  `live_vs_release_logits_exact` compare true live re-encoding to a release read
  (two records); the fresh child reports `cold_spans`, `cold_live_parity` and
  `cold_logits_parity` for two-record cold resume. Fresh output directories only;
  old runs are not overwritten.

Native work still outside this interface: durable read/write/retrieval bytes,
checkpoint artifact bytes and wall time. The v12 `cost` dictionaries report
them, but the workflow charges only source work to the v5 ledger; the native
operator must extend the packet plan before claiming complete accounting.
Physical SSD traffic and GPU-synchronized kernel time are not measured.

## Known limitations

1. CPU engineering evidence only; random weights cannot demonstrate useful
   memory or semantic recall.
2. Full-original parameter hashes are operator-declared in CPU fixtures; native
   admission must verify them before and after work. The provider binds exact
   adapter content, and a native operator should attest the already-verified
   content digest to avoid a second backbone hash.
3. The provider's runtime stamps detect in-place mutation and adapter/tokenizer
   contract substitution, but a tokenizer that changes behaviour without
   changing `identity()` is outside that guard.
4. The five core modules install one `SpinePort` per site. The separate density
   track now implements and tests the real-v12 matched two-group wrapper.
5. Pre-MLP fusion and 23/31 site sets are not measured here.
6. A native packet and guarded launch entry point are prepared separately but
   remain unreviewed, unsealed and unlaunched. Nothing here authorizes a launch.
7. Native uninterrupted-update equivalence is established only on the CPU
   fixture; native equivalence was not and cannot be claimed from this report.
8. The workflow's cap plan is an engineering envelope, not an authorization.

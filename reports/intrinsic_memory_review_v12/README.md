# Independent v12 correctness and evidence review

## Verdict

**Incorrect / changes required. The native packet is not ready, and the complete CPU engineering increment is not qualified.** The deterministic native restart budget failure alone blocks launch. The ordinary v12 happy-path CPU workflow is strong engineering evidence, but source-authority/cache defects and the successor/density contract failures prevent a blanket CPU qualification. No native model was imported or loaded by this review, no native attempt was launched, and no learned-capability or useful-recall claim is made.

Reviewed commits:

- `00edcad77ffeb59fa733bdf4aa217eb313b29179` — Replica successors
- `6dede67eb8d934898c2a83c451638a8cf71369c3` — contextual v12 producer
- `3cd683c34e9c9d7e8436b1c6c602833af3737f45` — density/cost
- `2b788c461bf0f1437906300383e6894609d6e4e8` — final native preparation

Final packet SHA-256: `3a2a7af1d08118df902e1dd20f5b3ca74b1c651005a2e0bf7f9eda7868198cc2`. Protocol SHA-256: `01d2e6d0fdd42265b8d48c765771f9dd2258c50ef7d2c552726e5976890b3a3f`; implementation receipt SHA-256: `0370704ba9c6467b6e49c37afd7684f6cdd3f6a493921210dd700556639008e5`.

The only pre-existing tracked dirty file observed was `reports/vision_overfit_20261025/helper_tests.xml`, SHA-256 `f43ba05ff6742682fad962bddd450610bf3ccffccacd0a48debf63647a64f9bb`; it was not touched. This review added only `tests/test_independent_review_v12_findings.py` (SHA-256 `24c4ffe92de7c3007c8e75ebc22fc6f6f40e4b165a5c13ec04cae9cf54689477`) and this report directory. No production file was edited.

## Confirmed findings, severity ordered

### P0 — launch/operations blocker

#### N1. Reserve the restart prompt-tokenization work actually required

- **Location:** `src/replica_lab/intrinsic_memory_coverage_v12.py:273-277`; consumers `src/replica_lab/intrinsic_memory_teacher_v12.py:171-199` and `src/replica_lab/intrinsic_memory_native_v12.py:451-462`.
- **Trigger/consequence:** restart restores 32 teacher targets. Each `expected(row)` performs student and exact-text encoding, requiring 64 prompt-tokenizer reservations, followed by three more for canary/update. The stage cap is four. The fifth reservation raises `BudgetError`; an authorized attempt would die after its fitting work, before restart equivalence/actual continuation evidence, with zero retries.
- **Reproduction:** `test_restart_prompt_budget_covers_teacher_revalidation_and_update` uses the real `plan("two")` and `WorkLedger`; it fails at the fifth of 67 required reservations.
- **Remedy:** derive and reserve at least 67 passes (and rebind packet/authorization/totals), or eliminate the repeated encoding with a contract-bound cache. Add a real restart-stage consumption test before preparing another packet.

### P1 — correctness, isolation, authority, or experimental-integrity defects

#### R1. Refuse cross-branch supersession and lineage

- **Location:** `src/replica_lab/successors/formation.py:977-995,1128-1153`.
- **Trigger/consequence:** a proposal on branch A can supersede or cite branch B without scope validation. The global tombstone then hides the branch-B record and can contaminate later lineage, violating branch isolation.
- **Reproduction:** `test_cross_branch_supersede_is_refused` commits a branch-B record and a branch-A child naming it; the invalid commit is accepted.
- **Remedy:** validate subject/branch/purpose/mode for every parent and ancestor before commit; define and authorize host-wide revocation explicitly. Add cross-branch lineage and revocation tests.

#### R2. Preserve external authority through cold resume and fallback

- **Location:** `src/replica_lab/successors/tiers.py:460-464,481-484,526-550`.
- **Trigger/consequence:** a cold ticket omits the authority callback and `resume()` passes `None`; an external correction/revocation during I/O therefore returns stale payload as `ready`. Fallback also lacks a post-read authority check.
- **Reproduction:** `test_cold_resume_revalidates_external_authority` changes authority while pending and observes `ready` with the stale bytes.
- **Remedy:** retain/re-supply authority at resume, revalidate after I/O, and apply the same check to fallback.

#### R3. Implement coherent P-clock advancement and generation selection

- **Location:** `src/replica_lab/successors/tiers.py:236-258,599-633`.
- **Trigger/consequence:** `publish_p` reuses `p_{owner}` but immutable record bytes contain the clock, so clock 2 after clock 1 raises instead of publishing. Same-payload republish with new metadata returns/caches the new generation while durable bytes retain the old generation; `read_p` also prefers warm without comparing authority/generation.
- **Reproduction:** `test_p_clock_can_advance` fails on the second publication with `TierError`.
- **Remedy:** use clock-versioned records plus an atomic current pointer, or an atomic monotonic replacement path; reconcile hot/durable metadata and select latest valid authority.

#### R4. Make chunk proposal, invalidation, and recomputation refer to the pre-chunk state

- **Location:** `src/replica_lab/successors/chunks.py:682-751`.
- **Trigger/consequence:** `recompute()` applies the update to the already-updated state, so it cannot reproduce the committed digest; a second `begin_chunk()` replaces the key while retaining the prior update; `invalidate()` clears only bookkeeping and retains discarded tensor effects. These break recomputation retry and exactly-once attribution.
- **Reproduction:** `test_chunk_recompute_reproduces_committed_digest`, `test_pending_chunk_cannot_be_replaced`, and `test_invalidate_restores_last_committed_state` all fail independently.
- **Remedy:** retain committed/pre-chunk state, reject a second begin while pending, restore committed state on invalidation, and compare recomputation to the actual receipt digest.

#### V1. Bind frozen-source authority to buffers as well as parameters

- **Location:** `src/replica_lab/contextual_source_features_v12.py:303-329`; export path `src/replica_lab/contextual_memory_layout_v12.py:304-331`.
- **Trigger/consequence:** in-place mutation of a non-parameter backbone buffer is invisible to `_original_stamp`; cached features remain valid and new work can be produced under stale source provenance. `save()` also does not revalidate the original before writing the extension.
- **Reproduction:** `test_buffer_only_source_mutation_invalidates_feature_authority` mutates a registered buffer and `validate_original()` incorrectly returns normally.
- **Remedy:** include named buffers in the mutation stamp or invoke the sealed full `base_unchanged()` audit, clear C on mismatch, and validate originals at export.

#### V2. Include actual source text in the tokenizer-cache identity

- **Location:** `src/replica_lab/contextual_source_features_v12.py:359-369,418-424`.
- **Trigger/consequence:** two spans with identical record geometry/hash metadata but different text share token-cache keys. The second span receives the first span token IDs, and that wrong sequence propagates into feature computation and provenance.
- **Reproduction:** `test_token_cache_binds_actual_canonical_text` obtains IDs for `badge tan` after `badge red`; it receives the red IDs.
- **Remedy:** bind cache entries to `text_digest`/the actual text and validate canonical text-to-hash consistency at the public boundary; add a same-geometry/different-text test.

#### F1. Keep a memory pin immutable under later tombstones

- **Location:** `src/replica_lab/successors/formation.py:1200-1220`.
- **Trigger/consequence:** correction/revocation after a pin changes what that same pin returns because tombstones are not gated by tombstone-event generation. A long interaction can lose previously read evidence mid-interaction.
- **Reproduction:** `test_pin_is_immutable_under_later_correction` observes the parent before correction and an empty set from the same pin afterward.
- **Remedy:** apply tombstones only when their event generation is visible to the pin, in addition to temporal filtering. If revocation is intentionally immediate/global, change the pin contract and controls rather than claiming immutable snapshots.

#### T1. Enforce producer, representation, scope, and validity on tier reads

- **Location:** `src/replica_lab/successors/tiers.py:302-324,413-520`.
- **Trigger/consequence:** publication stores these fields, but read/revalidation never checks them; expired or wrong-producer/representation/scope payloads can be returned `ready` unless every caller independently duplicates validation.
- **Reproduction:** static call-graph proof: `valid_from`/`valid_to` have no read-side consumer and `_read_tier` checks only payload identity/generation/authority revision.
- **Remedy:** require an expected read contract and cutoff (or mandatory authority that validates them) before and after I/O, with mismatch/expiry tests.

#### D1. Zero gradients for every declared cost-pilot update

- **Location:** `scripts/run_intrinsic_memory_density_v12_cost_pilot.py:59-104`.
- **Trigger/consequence:** updates accumulate gradients across passes, so the measured two/four-update schedule is not the declared whole-pass optimizer schedule. The timing may be similar, but exposure semantics and the fitted workload are mislabeled.
- **Reproduction:** `test_cost_pilot_zeroes_gradients_for_each_declared_update` observes zero `zero_grad` calls for two updates.
- **Remedy:** call `zero_grad(set_to_none=True)` before each pass and regenerate affected cost receipts under new destinations.

#### D2. Freeze numeric exposure selection in the validated protocol

- **Location:** `src/replica_lab/intrinsic_memory_density_protocols_v12.py:367-368,418-420`; `configs/measured_total_cost_v12.json:163-166`; pilot `:42-45,161-166`.
- **Trigger/consequence:** common updates `2` and cap `8` exist only as mutable module constants while the config contains placeholder strings. Changing the constants changes selection without changing the supposedly frozen protocol digest.
- **Reproduction:** static binding proof; `validate_protocol` validates the strings and the pilot never reads numeric values from the config.
- **Remedy:** put both numbers in the protocol, validate exact values, and have the pilot consume the validated protocol.

#### A1. Clear the actual in-memory previous workspace on task reset

- **Location:** `src/replica_lab/successors/adapters.py:333-338`; actual runtime field `src/replica_lab/runtime/loop.py:193,282-295`.
- **Trigger/consequence:** with no `WorkspaceStore`, reset writes nonexistent `_previous_workspace` instead of `previous_workspace`; the next decision can reuse old C while the receipt says it was cleared.
- **Reproduction:** static field/consumer proof; existing adapter tests always provide a store.
- **Remedy:** clear `runtime.previous_workspace` unconditionally and test the no-store intervention.

#### A2. Use the admitted source epoch for every formed item

- **Location:** `src/replica_lab/successors/adapters.py:293-314`; validation `src/replica_lab/successors/formation.py:844-875,1187-1189`.
- **Trigger/consequence:** the adapter hardcodes epoch 0; nonzero-epoch stores refuse it, and a proposal can contain observations/outcomes with a different epoch that commit but are unreadable under the pin.
- **Reproduction:** static proposal/record filter proof.
- **Remedy:** derive the epoch from runtime state and validate every observation/outcome against the pin/source admission.

#### S1. Derive study attribution from observed record identities

- **Location:** `src/replica_lab/successors/study.py:240-253`.
- **Trigger/consequence:** `improvement_attributable_to_write` is hardcoded true; `enabled_used_new_observation` checks only non-null, not the new record ID; provisional-note exclusion compares incompatible identifiers. The study can claim write-attributable improvement without establishing it.
- **Reproduction:** static consumer proof; these booleans are not computed from `new_id`/visible records.
- **Remedy:** require before/disabled/after identity comparisons and actual absence of provisional notes from evidence.

### P2 — edge-path correctness, control validity, and accounting defects

#### N2. Report non-termination as truncation

- **Location:** `src/replica_lab/intrinsic_memory_diagnostic_v12.py:323-327`; baseline gate `src/replica_lab/intrinsic_memory_native_v12.py:255-256`.
- **Trigger/consequence:** generation that exhausts eight tokens without EOS returns `truncated: False`, making the declared truncation observation/criterion unreliable.
- **Reproduction:** `test_nonterminating_generation_is_marked_truncated` returns `terminated=False, truncated=False`.
- **Remedy:** set `truncated = not stopped` and require termination explicitly in baseline competence.

#### N3. Re-derive tokenizer identity from the loaded tokenizer, not the declaration

- **Location:** `src/replica_lab/intrinsic_memory_native_v12.py:70-89,102-118`.
- **Trigger/consequence:** `BoundSourceTokenizer` is handed the declared identity and the subsequent digest compares that same mapping to its declared digest. The check is circular; normal launch remains protected by the independent full source-file hash loop, but this runtime check does not itself detect a substituted loaded tokenizer.
- **Reproduction:** static data-flow proof; changing only the wrapped tokenizer cannot change the digest.
- **Remedy:** derive/compare actual tokenizer files, vocab/config/template identity and loaded vocabulary properties.

#### V3. Restrict stale-release eviction to invalid records

- **Location:** `src/replica_lab/contextual_memory_v12.py:200-207`.
- **Trigger/consequence:** when one of two records has a stale producer release, `_batch` failure evicts both records from C, including the healthy current sibling. The next encode performs unplanned source work that may not have budget reservation.
- **Reproduction:** static mixed-release path proof; existing tests do not populate/assert the healthy sibling cache on this branch.
- **Remedy:** identify and evict only the failing record; test mixed releases with a healthy sibling.

#### V4. Bind release consumption to the declared source execution identity

- **Location:** `src/replica_lab/contextual_release_v12.py:241-334`.
- **Trigger/consequence:** a corpus published with `source_identity=A` can be consumed by a path declaring B when weights/contracts otherwise match; the manifest field is written but never checked.
- **Reproduction:** static authority-field proof.
- **Remedy:** compare manifest `source_identity` with `path.source_identity` at read/value/validation boundaries.

#### D3. Make profiler instrument registration atomic

- **Location:** `src/replica_lab/intrinsic_memory_density_v12.py:316-329,419-437`.
- **Trigger/consequence:** invalid active-site masks are rejected only after forward identity registration, poisoning retry and permitting recompute attribution to a nonexistent forward.
- **Reproduction:** `test_rejected_profiler_instrument_is_atomic` observes `f-9` in `forward_ids` after refusal.
- **Remedy:** validate all inputs before mutating identity maps and roll back earlier hooks if later profiler registration fails.

#### D4. Do not classify legitimate durable authority revalidation as a write side effect

- **Location:** `src/replica_lab/intrinsic_memory_density_v12.py:43-47,156-174`.
- **Trigger/consequence:** store-backed recompute increments `source_revalidations`, which is in `READ_WRITE_KEYS`; the assertion rejects the documented durable recompute path despite no retrieval/admission/snapshot/durable write.
- **Reproduction:** `test_durable_recompute_permits_authority_revalidation` raises on exactly that counter.
- **Remedy:** separate authority-validation metrics from prohibited side effects; report positive source recomputation rather than raising.

#### D5. Enforce the declared producer boundary API

- **Location:** `src/replica_lab/intrinsic_memory_density_v12.py:57-64,612-703`.
- **Trigger/consequence:** `PRODUCER_API` is unused; missing encoder/provider methods or a non-NativeFusedAdamW optimizer can pass construction and produce a null producer release instead of refusing a non-v12 boundary.
- **Reproduction:** static validation/data-flow proof.
- **Remedy:** enforce callable/API and optimizer/implementation identities before profiling.

#### F2. Apply pin cutoff and original order to formation reads

- **Location:** `src/replica_lab/successors/formation.py:1216-1220,1253-1266`; consumer `src/replica_lab/successors/study.py:40-50`.
- **Trigger/consequence:** `outcomes()` ignores cutoff/eligibility, exposing post-snapshot receipts; records within one event are ordered by hash-derived record ID rather than observation ordinal, so last-match selection can choose the wrong observation.
- **Reproduction:** static SQL/filter/consumer proof; no cutoff-outcome or same-query multi-observation test exists.
- **Remedy:** filter outcomes by the pin and persist/order by proposal ordinal.

#### A3. Execute and recover adapter interventions as declared

- **Location:** `src/replica_lab/successors/adapters.py:318-357`.
- **Trigger/consequence:** `identity_then_task` runs task first just like the reverse arm, so the order control is vacuous; exceptions during run/finish leave `_pin` installed and block every later interaction.
- **Reproduction:** static dispatch/control-flow proof.
- **Remedy:** dispatch identity-first explicitly, record/assert operation order, and clear pin/baseline in `finally`.

#### T2. Account movement and handle dirty/corrupt edge paths without false health

- **Location:** `src/replica_lab/successors/tiers.py:336-353,432-433,503-504,553-562,645-661`.
- **Trigger/consequence:** demote/promote discard actual write bytes; a dirty oldest hot entry aborts even when a later clean candidate exists; malformed generation raises raw `ValueError` while `recover()` can report the record healthy.
- **Reproduction:** static return-value, eviction-loop, and recovery-schema proof.
- **Remedy:** charge movement writes, skip protected dirty entries when clean candidates exist, validate manifest schema/types centrally, and quarantine invalid records.

#### C1. Enforce the chunk token bound at sequence creation

- **Location:** `src/replica_lab/successors/chunks.py:406-427`.
- **Trigger/consequence:** an initial prefix over 8,192 tokens is durably accepted, leaving a sequence that can never begin a valid chunk.
- **Reproduction:** `test_initial_prefix_respects_global_token_bound` observes no refusal.
- **Remedy:** validate length and token types in `open_sequence`.

### P3 — disabled-call compatibility

#### V5. Treat an empty active-site list like an empty tuple

- **Location:** `src/replica_lab/contextual_memory_layout_v12.py:256-260`.
- **Trigger/consequence:** `active_sites=[]` builds a snapshot and raises raw `ValueError` instead of skipping computation and preserving baseline output.
- **Reproduction:** `test_empty_list_active_sites_skips_memory_and_matches_base` reaches the reader and raises.
- **Remedy:** normalize site iterables before the disabled check and return the base path for every empty value.

## Suspected risks and explicit coverage gaps (not counted as confirmed passes)

- Two-record durable ordering ties (`contextual_memory_v12.py:131-133`) may make slot order retrieval-dependent; no multi-record live/warm/cold/release parity was established. Release-write failure charge rollback and whole-release cache eviction also need fault-injection coverage.
- `FormationStore.recover()` does not authenticate every derived tombstone/note/lineage table. P-specific revocation semantics and unverified-correction behavior under verified-only pins need an explicit contract; this review does not assume the desired policy.
- The density profiler has no post-update arm-distinguishing profile, negative controls, disjoint native pilot, or end-to-end native consumer. Existing density report files are not all reproducible from one checked-in command; this review did not treat their claimed totals as independent evidence.
- Native risks not behaviorally exercised here: output-directory-only attempt consumption, torn-tail append refusal, cached-ledger stat TOCTOU, uncharged source/base hashing and fit receipt writes, restart wall headroom, false-recall coverage, and partial native shared-library/wheel-origin binding. These remain open review items, not passes.
- The integrated CPU workflow release comparison is release-to-release, not a true live-versus-release check; cold tests largely assert shape/count rather than multi-record value parity. The workflow nevertheless independently passed actual fresh-process continuation.

## Independent evidence obtained

### Source-free / protected-absent conditions

Two detached worktrees were created with `git worktree add --no-checkout`, then non-cone sparse patterns `/*`, `!/src/replica_lab/runtime_pilot.py`, and `!/checkpoints/` were configured **before** checkout. Paths were `/private/tmp/replica-independent-v12-3cd683c3` and `/private/tmp/replica-independent-v12-2b788c46`. Post-checkout assertions confirmed the protected module and checkpoint tree were absent. Tests ran with `PYTHONDONTWRITEBYTECODE=1`; no weights were downloaded or copied. The separate native `--review` ran in the main tree only to hash the already-local source and validate closure; it did not import/load a model.

### Regression and native-focused suites

- At `3cd683c3`, inherited 32-file scope + matched-reader + all 12 new successor/v12/density files: **747 passed, 0 failed, 0 skipped**. Pytest 287.34 s; `/usr/bin/time` 287.82 s; max RSS **1,027,964,928 bytes**. Receipts: `regression_3cd683c3.{log,xml,time}`. This scope independently exercised the inherited r1/r2 latest-committed-hot versus cold-publication recovery, interrupted/missing/corrupt authority, revocation, and unrelated healthy-state protections.
- At `2b788c46`, final native files: **57 passed, 0 failed, 0 skipped**. Pytest 2.55 s; time 3.02 s; max RSS **376,832,000 bytes**. Receipts: `native_focused_2b788c46.{log,xml,time}`.
- Independent contract reproductions at `2b788c46`: **16 failed, 0 passed, 0 skipped** in 3.27 s; max RSS **436,502,528 bytes**. These are intentional defect demonstrations, not a passing qualification suite. Two earlier harness errors were corrected and are not counted as defect evidence. Final log/JUnit: `independent_reproductions_2b788c46.{log,xml,time}`.

Exact commands and file lists are recorded in `verification_summary.json` and `commands.md`.

### New integrated tiny-BF16 CPU workflow and actual continuation

Fresh output: `integrated_cpu_2b788c46/`. All 2/9/36 arms passed zero-init first-step behavior, subsequent producer/order/every-reader gradients, disabled computation skipping/output parity, masked padding invariance, live/durable parity, correction/revocation, cold pending recovery, complete extension reload, exact pre-logits/loss, one actual continuation update, and **exact uninterrupted weight and optimizer-moment comparison**. Original fixture parameters remained unchanged. `/usr/bin/time`: 17.62 s, max RSS 717,062,144 bytes; internal workflow wall 14.950200625 s; final artifact bytes **102,218,367**. This is a random tiny BF16/synthetic-tokenizer fixture, not Spark competence or native capability.

### Established checks at final commit

- Migration `--tracked require --no-write`: **1482 checks, 0 failed**, 11.06 s, RSS 31,244,288 bytes.
- Final product `--no-write`: **552/552**, 8.23 s, RSS 37,535,744 bytes.
- Final-product bundle `--check`: **293 inputs, 7,657,081 bytes**, digest `5be29a21022408aca555ba77284940ff42a734d3a4e9bd64909a34e09f1970d3`; 8.14 s, RSS 27,770,880 bytes.
- Artifact guard `--scope tracked`: **12,915 files**, largest 8,860,309 bytes, no violations; 0.17 s, RSS 34,848,768 bytes.
- Historical generated receipts were not overwritten; all new stdout/receipts are in this review directory.

### Native preflight, exposure, and launch status

Independent `--review` at `2b788c46` validated the exact packet above, **62 implementation files**, two source-execution files, and **8,238,996,247 source bytes**, with code verification before model loading. Independent `--environment-only` returned `environment_verified`, Python 3.12.14/Torch 2.14.1/Transformers 4.57.1/tokenizers 0.22.1, 12 critical files, `model_imported=false`, `native_execution=false`. The review took 4.18 s and 74,612,736-byte max RSS. These checks pass the static closure/environment gate but do not cure N1-N3.

The v11 development exposure receipt remains `development_opened=false`; v12 admission reports development labels unparsed. This review hashed prepared data only and did not open development labels. No authorization receipt was created, no native attempt was consumed, and launch remains blocked. Historical v11 fitting failure remains a failure; no useful native recall, P learning, promotion, or RSI is established.

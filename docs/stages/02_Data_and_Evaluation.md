# S1 — causal data intake and independent evaluation

Reviewed **22 October 2026, America/Chicago**. See the [master roadmap](00_Master_Roadmap.md), [current execution state](../../execution_state.md) and [dated primary-source review](07_Research_Evidence_20261022.md). Gates and budgets are proposed unless a linked existing report establishes execution. Planned artifacts below do not claim files already exist.

The current fictional fixtures are useful contracts, not a representative training corpus. Start from deterministic source worlds and independent solvers; only then verbalize cases or ingest one approved public source. Keep the rich canonical source/episode manifest outside the minimal eight-field export in the dataset guide.

Roles are train, dev, calibration, promotion and audit. Learning-policy reward also needs its own exposure boundary. Split original sources/worlds/personas/decision families before all teacher derivatives. An observed source can naturally contain the needed fact; an oracle answer/evidence-location annotation copied into memory is leakage. Event time and observation time are separate.

The bounded S1.1 contract preserves immutable causal episodes, explicit-purpose exports, source/event lineage, corrections, tombstones and current permission revocation. S1.2 now provides a **2,000-target, 1,000-world solver-backed synthetic infrastructure release**, extending the preserved [20-episode preparation](../experiments/20261022_Episode_Slice.md). Four shared mechanisms contribute 500 targets each; roles contain 1,200 train and 200 each dev/calibration/promotion/audit. All roles are exposed contract material. World separation supports unseen-world checks, not unseen-mechanism transfer or a sealed final audit. [Contract design](../experiments/episode_contract_design.md), [v2 release and preflight](../experiments/20261022_Infrastructure_Slice_V2.md).

Offline native tokenization admits all 2,000 targets under 512 with **zero truncation**, maximum 334 full-sequence tokens and 501,800 native tokens across all roles. The train export contains 300,369 native tokens and 4,374 supervised answer/EOS tokens; an augmented source writer adds a separate 233,295-token view. These are single-pass dataset counts, not a campaign budget or total model compute. [Token accounting](../../reports/infrastructure_preflight_v2_20261022/token_accounting.json).

S1.4 preserves the original frozen B1 **0/16 task-correct, 0/16 format-valid, all-capped** negative. The predeclared one-line contract v1 rerun keeps the same model, sixteen cases, parser and 32-token cap: **16/16 single-line responses and EOS, zero caps, but only 1/16 grammar-valid and 0/16 task-correct**. Fifteen outputs combine incompatible response forms; the one valid ask incorrectly requests fields already supplied. Evidence-use and conditional format selection remain open failures, so budget alone cannot explain the result. [Frozen protocol](../experiments/20261022_Response_Contract_V1_Protocol.md), [results and diagnosis](../experiments/20261022_Response_Contract_V1_Results.md).

The parser remains unchanged. Retention, calibration and model evidence-sensitivity controls remain incomplete. S1.3 has a [pinned metadata review](../experiments/20261022_Public_Source_Admission_Review.md), with both public candidates quarantined. S1.5's 20k release and 8–32M processed-token budget remain proposed caps. Independent public admission and locked evaluation still block the [S2.4 screen proposal](../experiments/20261022_S2_4_Screen_Proposal.md).

## S1.1 — Freeze provenance, roles and causal episode contracts

**Status:** `bounded_contract_implemented`. **Dependencies:** none; can start independently.

Implemented a strict immutable canonical episode and deterministic eight-field SFT export with source/group/split binding, event versus observation time, current use/privacy permissions, source versus agent ownership, explicit oracle separation, late corrections and tombstones. Complete-release auditing binds declared duplicate source bytes to one family/split and preserves event bytes. The bounded contract has 47 passing targeted CPU tests; it is not a data factory.

- **Artifact:** src/replica_lab/episodes.py; tests/test_episodes.py; schemas/episodes/episode.schema.json; docs/experiments/episode_contract_design.md.
- **Interface:** Trusted intake -> Episode/validate_episode -> complete-release audit_episode_families -> host-reviewed verifier/byte-bound VerificationReceipt -> input_view/export_sft. Target-bearing purposes are training/evaluation/teacher only; audit requires explicit local_evaluation. Preserve rich source/receipt/export-purpose manifest beside lossy SFT.
- **Test gate:** Implemented and tested structural/temporal/ownership/purpose boundaries, explicit oracle exclusion, source-hash family binding, immutable event history and verifier receipt binding. Source hashes/authority/consent are trusted declarations, not raw-byte authentication. Ordinary observations may legitimately contain answer facts; semantic gold laundering requires trusted adapter/verifier review. No public corpus or 2k release is admitted by these checks.
- **Resources:** CPU-only 47 targeted tests and one documented fictional late-correction fixture; no authentic biography, downloaded data or model execution. Three top-level schemas remain unchanged; the episode schema is nested.
- **Failure policy:** Quarantine unsupported or ambiguous records; never repair missing biography/license/split information by invention.
- **Next smallest task:** Preserve these contracts while extending source-specific public intake and independent evaluation; use the new S1.2 release without promoting exposed fixture roles to sealed evidence.

## S1.2 — Release a 2,000-target solver-backed infrastructure slice

**Status:** `bounded_2000_target_infrastructure_release`. **Dependencies:** S1.1.

Released 2,000 targets from 1,000 complete synthetic worlds: register XOR, affine modulo 97, finite-state transitions and bounded tool-recovery decisions, 500 targets per mechanism. Roles split worlds before derivation; all four mechanisms and the grammar remain shared. Every role contains 50% answer, 25% ask and 25% unknown cases across late correction, late-observation and tombstone scenarios. Tool-recovery labels are solver-defined proposed decisions, not measured execution of a live tool loop.

The host compiler's `input_view` selects only active eligible observations after chronology, correction and tombstone processing. Native evaluation therefore measures reasoning/actions over that compiled snapshot. It does not measure model-owned temporal retrieval, correction detection or state reconstruction from the full raw history.

- **Artifact:** data/diagnostics/infrastructure_slice_v2/manifest.json; scripts/make_infrastructure_slice.py; src/replica_lab/infrastructure_solver.py; tests/test_infrastructure_slice.py; tests/test_infrastructure_solver.py; reports/infrastructure_preflight_v2_20261022/stratified_review.json; reports/infrastructure_preflight_v2_20261022/token_accounting.json; [release record](../experiments/20261022_Infrastructure_Slice_V2.md). The v1 slice and its historical evidence remain preserved.
- **Interface:** World generator -> executable oracle -> optional teacher candidates (verified false) -> independent validator -> admitted exports.
- **Test gate:** All 2,000 raw-source solver/chronology and export checks pass. Semantic-duplicate rejection retries 59 candidates and admits zero duplicate world signatures. Automated stratified review covers 320 episodes/160 worlds; this is an operational software review, not a human annotation study or proof of zero semantic error. Tokenization admits all cases under 512 without truncation. Public exposed audit/promotion packets are not sealed benchmarks; no held-out-mechanism or broad-retention claim follows.
- **Resources:** Host solvers and the existing offline native tokenizer; no teacher, public corpus, authentic biography or training. Single-pass native totals are 501,800 all-role / 300,369 train tokens; added source-writer work is reported separately.
- **Failure policy:** Fix the generator/solver or drop a failing category. Preserve rejection samples and recompute derived hashes; avoid scaling defective examples.
- **Next smallest task:** Use the frozen release for bounded S1.4 development evaluation; resolve public admission, independent retention/calibration cases and the S1.5 budget/exposure plan before a training campaign.

## S1.3 — Qualify one public training adapter and independent benchmark

**Status:** `metadata_review_quarantined_pending_adapter`. **Dependencies:** S1.1.

Resolve immutable PersonaMem-v2 card/data revision and training identities, license scope and causal fields; build one training adapter. Separately pin cleaned LongMemEval S with evaluator-only gold/evidence fields. Do not bulk-ingest the research catalog.

- **Artifact:** [metadata admission review](../experiments/20261022_Public_Source_Admission_Review.md); reports/public_source_metadata_review_20261022.json. Approved subsets and source-specific adapters remain planned.
- **Interface:** Pinned upstream source -> source-specific adapter -> canonical episodes; official benchmark labels never enter training, retrieval or teacher inputs.
- **Test gate:** Card/license/file hashes and identity-group split checks recorded; train/benchmark identities and cross-modal overlap audited. Oracle retrieval is a labelled evaluator condition, not ordinary input access.
- **Resources:** Official metadata/cards/code only; no corpus or model files downloaded. PersonaMem and cleaned LongMemEval revisions are pinned; payload hashes remain advertised metadata until actual byte verification. Long histories require declared retrieval/chunk budgets.
- **Failure policy:** Leave source quarantined if terms, identity overlap or causal cutoff is unresolved. A Hugging Face auto-detected train split is not permission to train on benchmark targets.
- **Next smallest task:** Resolve PersonaHub inherited terms and component attribution, then specify minimal pinned payload inspection. Audit complete linked pairs, six-split persona overlap, duplicate history versions and missing filename ID 558 before source-specific adapter admission.

## S1.4 — Lock generation, retention, calibration and evidence controls

**Status:** `partial_response_contract_v1_residual_format_semantic_failure`. **Dependencies:** S1.1, S0.3.

Extend strict generation evaluation beyond XOR: answer/ask/tool actions, temporal correction and held-out procedural tasks. Lock parsers, budgets, seeds, prompt/template/thinking settings and B0/B1 evidence visibility. Add constant-label, shuffled/missing evidence and matched inference controls.

- **Artifact:** src/replica_lab/infrastructure_scoring.py; scripts/prepare_infrastructure_preflight.py; tests/test_infrastructure_scoring.py; reports/infrastructure_preflight_v2_20261022/evaluation_spec.json and preserved adjacent data/control/native results; [original preflight record](../experiments/20261022_Infrastructure_Slice_V2.md). Contract v1 adds configs/response_contract_v1.json, src/replica_lab/response_contract.py, scripts/prepare_response_contract.py, scripts/score_response_contract_generations.py and the [paired raw score](../../reports/response_contract_v1_20261022/native_B1_scoring.json), [frozen protocol](../experiments/20261022_Response_Contract_V1_Protocol.md) and [result record](../experiments/20261022_Response_Contract_V1_Results.md). Full retention/calibration/control score matrices remain planned.
- **Interface:** Role-separated case manifest -> model generation -> fixed parser/solver -> per-family scores and paired/clustered uncertainty. Calibration thresholds use their own split.
- **Test gate:** Strict whole-response parser and independent solver classify answer/ask/unknown/tool-recovery targets. CPU constant-answer controls reject perfect-fit collapse; best constant is UNKNOWN at 50/200 dev. The preserved original B1 scores 0/16 task-correct and format-valid with all 16 capped and no EOS. Contract v1 produces 16/16 single-line responses and EOS with zero caps, but only 1/16 grammar/contract-valid and 0/16 task-correct: 15 format errors and one wrong missing-field value. Its 16/16 grammar/single-line/EOS gate fails. Mixed UNKNOWN/field-list responses and an ask for already observed RULE/SYMBOLS show residual conditional-format and evidence-use errors; they do not isolate a causal sentence or prove general semantic inability. Parser rules and cap remain fixed. The code suite passes 671 tests with one restricted-context MPS skip, separately from native task failure. Semantic missing-fact checks remain evaluator controls, not model sensitivity. Model evidence comparisons, calibration, retention and matched architecture compute remain pending.
- **Resources:** One contract-v1 frozen-original B1 job on the same 16 ordered dev worlds/four shared mechanisms, greedy cap 32, ceiling 512, seed 17, MPS FP32/fallback disabled. Protocol commit `430e298` precedes generation; no prompt, cap or parser tuning followed the result. Exact guard replay passes with zero observed swap growth. Reused the local original with zero downloads, optimizer updates or new checkpoints. The original failed run and its archive remain intact; the dated result records own raw generation, token and memory evidence.
- **Failure policy:** Preserve raw failures and the unchanged parser. Predeclare any new development prompt or budget before another run; do not reinterpret exposed cases as sealed evidence or promote this result.
- **Next smallest task:** Predeclare a separate v2 development protocol with a query-policy-specific two-branch grammar to remove the ambiguous OR and discourage placeholder copying. Select the policy from public query text only, never gold action/value or observed missingness. Keep this v1 negative intact, then qualify B0/B1, shuffled/missing-evidence and independent retention/calibration controls before training comparisons. S1.4 remains partial.

## S1.5 — Predeclare and release the 20,000-target pilot

**Status:** `pending`. **Dependencies:** S1.2, S1.3, S1.4, S0.5.

After the infrastructure slice passes, admit the proposed source-allowlisted mixture: 8k general replay, 3k reasoning, 4k temporal, 2k generic personalization, 1k tool recovery and 2k ask/retrieve/verify/stop. Reconcile complete target/token counts with actual measured capacity.

- **Artifact:** planned: immutable pilot release and subset/license manifest; planned: predeclared comparison/power/exposure plan.
- **Interface:** Approved source manifests + verified generators -> grouped immutable release consumed identically by D0/D1/D2; all interface warmup and teacher lineage retained.
- **Test gate:** Freeze roles, family counts, processed/support/supervised token accounting, cap and statistical units before campaign. Proposed 8–32M processed-token budget must fit measured hardware/lengths and include failed/search runs in costs.
- **Resources:** Start one-seed short screen; scale only shortlisted conditions. 20k is a proposed cap, not a mandate to fabricate missing eligible data.
- **Failure policy:** Reduce the admitted pilot transparently if sources are unavailable; do not silently fill gaps with correlated paraphrases or benchmark targets.
- **Next smallest task:** Use the completed 2k infrastructure quality/length accounting to draft the actual mixture and compute budget; finish S1.3/S1.4 admission and grouped exposure/power analysis before releasing additional pilot data or starting S2.4.

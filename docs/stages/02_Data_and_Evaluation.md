# S1 — causal data intake and independent evaluation

Reviewed **22 October 2026, America/Chicago**. See the [master roadmap](00_Master_Roadmap.md), [current execution state](../../execution_state.md) and [dated primary-source review](07_Research_Evidence_20261022.md). Gates and budgets are proposed unless a linked existing report establishes execution. Planned artifacts below do not claim files already exist.

The current fictional fixtures are useful contracts, not a representative training corpus. Start from deterministic source worlds and independent solvers; only then verbalize cases or ingest one approved public source. Keep the rich canonical source/episode manifest outside the minimal eight-field export in the dataset guide.

Roles are train, dev, calibration, promotion and audit. Learning-policy reward also needs its own exposure boundary. Split original sources/worlds/personas/decision families before all teacher derivatives. An observed source can naturally contain the needed fact; an oracle answer/evidence-location annotation copied into memory is leakage. Event time and observation time are separate.

The bounded S1.1 contract preserves immutable causal episodes, explicit-purpose exports, source/event lineage, corrections, tombstones and current permission revocation. S1.2 now provides a **2,000-target, 1,000-world solver-backed synthetic infrastructure release**, extending the preserved [20-episode preparation](../experiments/20261022_Episode_Slice.md). Four shared mechanisms contribute 500 targets each; roles contain 1,200 train and 200 each dev/calibration/promotion/audit. All roles are exposed contract material. World separation supports unseen-world checks, not unseen-mechanism transfer or a sealed final audit. [Contract design](../experiments/episode_contract_design.md), [v2 release and preflight](../experiments/20261022_Infrastructure_Slice_V2.md).

Offline native tokenization admits all 2,000 targets under 512 with **zero truncation**, maximum 334 full-sequence tokens and 501,800 native tokens across all roles. The train export contains 300,369 native tokens and 4,374 supervised answer/EOS tokens; an augmented source writer adds a separate 233,295-token view. These are single-pass dataset counts, not a campaign budget or total model compute. [Token accounting](../../reports/infrastructure_preflight_v2_20261022/token_accounting.json).

S1.4 preserves the original all-capped **0/16** B1 and contract-v1 **0/16** negatives. The [predeclared query-policy v2](../experiments/20261022_Response_Policy_V2_Protocol.md), committed as `b2010c2`, selects one missing-response fallback from public query text only. The same sixteen-case M2 Ultra GPU probe now gives **4/16 strict-correct**, **12/16 legal grammar forms**, and **16/16 single-line responses with EOS and zero caps**. A fresh-context subagent independently judges **4/16 semantically correct and 4/16 instruction-compliant**. Program grammar and agent instruction compliance are different metrics. All four successes are UNKNOWN, exactly the selected constant-UNKNOWN control; all eight complete-evidence answers fail, and four asks request already observed fields. This is abstention collapse, not useful reasoning or architecture gain. [Results and error analysis](../experiments/20261022_Response_Policy_V2_Results.md), [strict report](../../reports/response_policy_v2_20261022/native_B1_scoring.json), [raw semantic judgments](../../reports/response_policy_v2_20261022/judge_semantic_raw.json).

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

**Status:** `partial_query_policy_v2_abstention_collapse`. **Dependencies:** S1.1, S0.3.

Extend strict generation evaluation beyond XOR: answer/ask/tool actions, temporal correction and held-out procedural tasks. Lock parsers, budgets, seeds, prompt/template/thinking settings and B0/B1 evidence visibility. Add constant-label, shuffled/missing evidence and matched inference controls.

- **Artifact:** The original [preflight record](../experiments/20261022_Infrastructure_Slice_V2.md), [v1 protocol](../experiments/20261022_Response_Contract_V1_Protocol.md) and [v1 results](../experiments/20261022_Response_Contract_V1_Results.md) remain unchanged. V2 adds configs/response_policy_contract_v2.json, configs/response_policy_judge_rubric_v1.json, src/replica_lab/response_policy_contract.py, scripts/prepare_response_policy_contract.py and scripts/score_response_policy_generations.py, with a [predeclared protocol](../experiments/20261022_Response_Policy_V2_Protocol.md), [results](../experiments/20261022_Response_Policy_V2_Results.md), [strict scoring](../../reports/response_policy_v2_20261022/native_B1_scoring.json) and [raw independent semantic review](../../reports/response_policy_v2_20261022/judge_semantic_raw.json). Full retention/calibration/control score matrices remain planned.
- **Interface:** Role-separated case manifest -> model generation -> unchanged parser/raw-source solver -> per-family scores and paired/clustered uncertainty. A separately predeclared fresh-context agent receives public query/evidence/completion without gold or fixed grades; preserve semantic judgments and reconcile them separately. Calibration thresholds use their own split.
- **Test gate:** V2 yields 4/16 strict-correct, 12/16 legal grammar forms, 16/16 single-line/EOS and zero caps. Independent semantic review yields 4/16 correct and 4/16 instruction-compliant. The agent assesses requested policy and evidence use; the program grammar metric checks legal forms, so those counts must not be conflated. All correct cases are UNKNOWN, matching the selected dev16 constant control at 4/16; the all-dev constant ceiling remains 50/200. All eight complete-evidence answers and all four ask cases fail. The format gate fails. The predeclared suite passes 798 tests/one restricted-context MPS skip; final integration including 21 archive-reader tests passes **819 tests/one skip**. Both are software evidence, separate from native task acceptance. Model evidence-sensitivity comparisons, calibration, retention and matched architecture compute remain pending.
- **Resources:** One frozen-original M2 Ultra MPS FP32 job after predeclaration commit `b2010c2`; same sixteen ordered worlds, evidence/labels, native template, greedy cap 32, ceiling 512 and seed 17, CPU fallback disabled. Query policy is a public instruction input, never an expected-action or missingness input. The original/v1 failures are preserved. Zero downloads, optimizer updates or new checkpoints.
- **Failure policy:** Preserve raw outputs, all three negative records and the unchanged parser. The supplemental archived-guard reader passes byte-bound historical path resolution and exact guard replay; frozen generation and scoring contracts remain unchanged. Predeclare separate interventions before another native run; no post-result tuning under v2, sealed-evidence relabelling or promotion.
- **Next smallest task:** Separately predeclare source-grounding/input-projection diagnosis and upstream sampling/thinking qualification, keeping other conditions fixed rather than changing every setting together. Then qualify B0/B1, shuffled/missing-evidence and independent retention/calibration controls before training comparisons. Independent early S3.1 may qualify a fixed-original MPS proposer and actual host tool receipts under its own protocol; see the [bounded host action preparation](../experiments/20261022_Host_Action_Contract.md). S1.4 remains partial, S1.3/S1.5 pending and S2.4 blocked.

## S1.5 — Predeclare and release the 20,000-target pilot

**Status:** `pending`. **Dependencies:** S1.2, S1.3, S1.4, S0.5.

After the infrastructure slice passes, admit the proposed source-allowlisted mixture: 8k general replay, 3k reasoning, 4k temporal, 2k generic personalization, 1k tool recovery and 2k ask/retrieve/verify/stop. Reconcile complete target/token counts with actual measured capacity.

- **Artifact:** planned: immutable pilot release and subset/license manifest; planned: predeclared comparison/power/exposure plan.
- **Interface:** Approved source manifests + verified generators -> grouped immutable release consumed identically by D0/D1/D2; all interface warmup and teacher lineage retained.
- **Test gate:** Freeze roles, family counts, processed/support/supervised token accounting, cap and statistical units before campaign. Proposed 8–32M processed-token budget must fit measured hardware/lengths and include failed/search runs in costs.
- **Resources:** Start one-seed short screen; scale only shortlisted conditions. 20k is a proposed cap, not a mandate to fabricate missing eligible data.
- **Failure policy:** Reduce the admitted pilot transparently if sources are unavailable; do not silently fill gaps with correlated paraphrases or benchmark targets.
- **Next smallest task:** Use the completed 2k infrastructure quality/length accounting to draft the actual mixture and compute budget; finish S1.3/S1.4 admission and grouped exposure/power analysis before releasing additional pilot data or starting S2.4.

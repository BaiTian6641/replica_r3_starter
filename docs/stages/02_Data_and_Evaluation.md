# S1 — causal data intake and independent evaluation

Reviewed **22 October 2026, America/Chicago**. See the [master roadmap](00_Master_Roadmap.md), [current execution state](../../execution_state.md) and [dated primary-source review](07_Research_Evidence_20261022.md). Gates and budgets are proposed unless a linked existing report establishes execution. Planned artifacts below do not claim files already exist.

The current fictional fixtures are useful contracts, not a representative training corpus. Start from deterministic source worlds and independent solvers; only then verbalize cases or ingest one approved public source. Keep the rich canonical source/episode manifest outside the minimal eight-field export in the dataset guide.

Roles are train, dev, calibration, promotion and audit. Learning-policy reward also needs its own exposure boundary. Split original sources/worlds/personas/decision families before all teacher derivatives. An observed source can naturally contain the needed fact; an oracle answer/evidence-location annotation copied into memory is leakage. Event time and observation time are separate.

The bounded S1.1 contract is implemented with **47 targeted CPU tests**: immutable causal episodes, explicit-purpose export, audit-only evaluation access, source/event lineage checks, late corrections, tombstones and current permission revocation. Source metadata/hash authority remains trusted intake, and semantic gold laundering is not detected automatically. Preserve canonical provenance, receipts and declared export purpose beside flattened SFT. This supplies the interface for S1.2; no CLI, public ingestion or 2,000-target factory is claimed. [Contract design and exact example](../experiments/episode_contract_design.md).

The 2k/20k counts and 8–32M processed-token budget are proposed caps. Admission depends on exact source terms, complete response lengths and actual Mac profiles. An external benchmark is not a curriculum pool. Development generations diagnose formatting, cap failures and constant answers; promotion is an independently controlled selection process; final audit is sealed until final reporting.

## S1.1 — Freeze provenance, roles and causal episode contracts

**Status:** `bounded_contract_implemented`. **Dependencies:** none; can start independently.

Implemented a strict immutable canonical episode and deterministic eight-field SFT export with source/group/split binding, event versus observation time, current use/privacy permissions, source versus agent ownership, explicit oracle separation, late corrections and tombstones. Complete-release auditing binds declared duplicate source bytes to one family/split and preserves event bytes. The bounded contract has 47 passing targeted CPU tests; it is not a data factory.

- **Artifact:** src/replica_lab/episodes.py; tests/test_episodes.py; schemas/episodes/episode.schema.json; docs/experiments/episode_contract_design.md.
- **Interface:** Trusted intake -> Episode/validate_episode -> complete-release audit_episode_families -> host-reviewed verifier/byte-bound VerificationReceipt -> input_view/export_sft. Target-bearing purposes are training/evaluation/teacher only; audit requires explicit local_evaluation. Preserve rich source/receipt/export-purpose manifest beside lossy SFT.
- **Test gate:** Implemented and tested structural/temporal/ownership/purpose boundaries, explicit oracle exclusion, source-hash family binding, immutable event history and verifier receipt binding. Source hashes/authority/consent are trusted declarations, not raw-byte authentication. Ordinary observations may legitimately contain answer facts; semantic gold laundering requires trusted adapter/verifier review. No public corpus or 2k release is admitted by these checks.
- **Resources:** CPU-only 47 targeted tests and one documented fictional late-correction fixture; no authentic biography, downloaded data or model execution. Three top-level schemas remain unchanged; the episode schema is nested.
- **Failure policy:** Quarantine unsupported or ambiguous records; never repair missing biography/license/split information by invention.
- **Next smallest task:** Build S1.2 source-bound synthetic packet/register worlds and a separately reviewed oracle, audit a small complete slice, then scale toward the proposed 2,000 targets.

## S1.2 — Release a 2,000-target solver-backed infrastructure slice

**Status:** `pending`. **Dependencies:** S1.1.

Generate independent microengineering worlds plus temporal correction, unknown/ask and tool-recovery cases. Compute answers/actions with solvers before teacher verbalization. Balance target/action categories and split complete worlds/rule families before derivatives.

- **Artifact:** planned: data release manifest with generator/solver/code hashes; planned: 2,000-target train/dev exports and stratified review report.
- **Interface:** World generator -> executable oracle -> optional teacher candidates (verified false) -> independent validator -> admitted exports.
- **Test gate:** All admitted labels pass their independent verifier; no duplicate world across roles; target/length/action histograms and rejected cases reported. Review at least 20 cases in each critical category as an operational sample, not proof of zero error.
- **Resources:** CPU solvers first; external teacher only with eligible data and logged model/prompt/cost. No audit material or real personal data sent by default.
- **Failure policy:** Fix the generator/solver or drop a failing category. Preserve rejection samples and recompute derived hashes; avoid scaling defective examples.
- **Next smallest task:** Implement a source-bound synthetic packet/register family with independent executable oracle, explicit raw-source hashes, causal unknown/ask/correction cases and complete-release lineage audit; inspect a small slice before scaling to 2,000.

## S1.3 — Qualify one public training adapter and independent benchmark

**Status:** `pending`. **Dependencies:** S1.1.

Resolve immutable PersonaMem-v2 card/data revision and training identities, license scope and causal fields; build one training adapter. Separately pin cleaned LongMemEval S with evaluator-only gold/evidence fields. Do not bulk-ingest the research catalog.

- **Artifact:** planned: pinned public-source manifest and approved subset list; planned: PersonaMem adapter and LongMemEval evaluation adapter audits.
- **Interface:** Pinned upstream source -> source-specific adapter -> canonical episodes; official benchmark labels never enter training, retrieval or teacher inputs.
- **Test gate:** Card/license/file hashes and identity-group split checks recorded; train/benchmark identities and cross-modal overlap audited. Oracle retrieval is a labelled evaluator condition, not ordinary input access.
- **Resources:** CPU inspection and minimal selected files; no model needed. Long histories use declared retrieval/chunk budgets rather than silent truncation.
- **Failure policy:** Leave source quarantined if terms, identity overlap or causal cutoff is unresolved. A Hugging Face auto-detected train split is not permission to train on benchmark targets.
- **Next smallest task:** Resolve one PersonaMem-v2 immutable revision and inspect complete history/query pairs before designing the adapter.

## S1.4 — Lock generation, retention, calibration and evidence controls

**Status:** `pending`. **Dependencies:** S1.1, S0.3.

Extend strict generation evaluation beyond XOR: answer/ask/tool actions, temporal correction and held-out procedural tasks. Lock parsers, budgets, seeds, prompt/template/thinking settings and B0/B1 evidence visibility. Add constant-label, shuffled/missing evidence and matched inference controls.

- **Artifact:** planned: versioned evaluation specification and case manifest; planned: baseline/retention/control score reports with raw generations.
- **Interface:** Role-separated case manifest -> model generation -> fixed parser/solver -> per-family scores and paired/clustered uncertainty. Calibration thresholds use their own split.
- **Test gate:** Raw completions, caps/EOS, format failures and answer correctness preserved. Evaluator cannot access training internals and candidate cannot alter oracle. Distinguish overfit cases from unseen inputs and unseen task families.
- **Resources:** CPU evaluator contracts, then bounded MPS baseline generation. Freeze inference cap before comparison and report token/time costs.
- **Failure policy:** Fix parser or case validity on development only; version the changed evaluator. Exposed audit cases are retired from sealed claims.
- **Next smallest task:** Add a constant-answer rejection check and classify format/abstention errors independently from solver correctness.

## S1.5 — Predeclare and release the 20,000-target pilot

**Status:** `pending`. **Dependencies:** S1.2, S1.3, S1.4, S0.5.

After the infrastructure slice passes, admit the proposed source-allowlisted mixture: 8k general replay, 3k reasoning, 4k temporal, 2k generic personalization, 1k tool recovery and 2k ask/retrieve/verify/stop. Reconcile complete target/token counts with actual measured capacity.

- **Artifact:** planned: immutable pilot release and subset/license manifest; planned: predeclared comparison/power/exposure plan.
- **Interface:** Approved source manifests + verified generators -> grouped immutable release consumed identically by D0/D1/D2; all interface warmup and teacher lineage retained.
- **Test gate:** Freeze roles, family counts, processed/support/supervised token accounting, cap and statistical units before campaign. Proposed 8–32M processed-token budget must fit measured hardware/lengths and include failed/search runs in costs.
- **Resources:** Start one-seed short screen; scale only shortlisted conditions. 20k is a proposed cap, not a mandate to fabricate missing eligible data.
- **Failure policy:** Reduce the admitted pilot transparently if sources are unavailable; do not silently fill gaps with correlated paraphrases or benchmark targets.
- **Next smallest task:** Produce a 2k-slice quality/length report and budget spreadsheet before admitting the remaining pilot data.

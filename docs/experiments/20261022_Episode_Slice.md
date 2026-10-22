# S1.2 preparation: source-bound register slice

**22 October 2026, America/Chicago.** This is a small **synthetic infrastructure release**, not the proposed 2,000-target dataset or a model-training result. It uses no teacher, network, model, tokenizer, personal archive or authentic identity material. Existing native eight-case datasets are unchanged.

The release is [episode_slice_v1](../../data/diagnostics/episode_slice_v1/manifest.json): **10 world groups, 20 canonical queries**, with two worlds and four queries in each role. Every world has one subject, source branch, immutable source artifact and two related queries. Role assignment is archived **before** generating world events or targets. Queries from one world remain in that world's group/role; they are not independent statistical samples.

| Role | Worlds | Queries | Flattened SFT export |
|---|---:|---:|---|
| train | 2 | 4 | `exports/train.jsonl`, local-training purpose |
| dev | 2 | 4 | `exports/dev.jsonl`, local-evaluation purpose |
| calibration | 2 | 4 | None; evaluator-only canonical packet |
| promotion | 2 | 4 | None; evaluator-only canonical packet |
| audit | 2 | 4 | None; evaluator-only canonical packet |

These public synthetic audit/promotion fixtures are **exposed contract/test material, not a sealed audit or an independent promotion benchmark**. They were generated and inspected for contract validation. No audit/calibration/promotion targets enter the train/dev flattened exports. Default audit retrieval is denied by the episode compiler; explicit local evaluation still requires a trusted evaluator/host. File layout and permission metadata are not an operating-system access-control service.

## Tasks and chronology

The known rule family computes `INPUT XOR MASK` over 8-bit synthetic registers. Each role contains one late-correction world and one late-observation/deletion world:

- In the correction world, the first query precedes a correction's **observation** time and the second follows it. Both numeric answers are independently checked and differ. An old event timestamp does not make a later report available earlier.
- In the other world, MASK is unavailable at the first cutoff, then is observed and later tombstoned. The two queries explicitly instruct `ask` versus `UNKNOWN` when a register is unavailable. That instruction is visible in the question; no hidden deletion-status label is needed to choose the response. INPUT remains eligible evidence.

Each role therefore contains two numeric-answer, one ask and one unknown target. All roles use the same XOR rule family and question grammar. Disjoint world/source identities prevent direct reuse of raw source bytes across roles; they do not establish task-family generalization or independent benchmark statistics. This is not an answer-balanced training campaign or an unseen-rule-family benchmark. The synthetic sources contain natural answer-bearing register facts; those are permitted observations. Evaluator metadata and the target are never included in `input_view` memory. The existing compiler excludes future/superseded/deleted observations and enforces current purpose permissions.

## Raw-source and derivation binding

[make_episode_slice.py](../../scripts/make_episode_slice.py) archives exact UTF-8 source text under `raw_sources/`: a synthetic-world header followed by chronological event JSON lines. [source_manifest.json](../../data/diagnostics/episode_slice_v1/source_manifest.json) records each file's SHA256/byte length and every event's half-open byte span plus span hash. Source headers retain role/group/subject/branch/actor, explicit synthetic classification and allowed purposes. [world_roles.json](../../data/diagnostics/episode_slice_v1/world_roles.json) is the pre-derivation assignment.

[derivation_manifest.json](../../data/diagnostics/episode_slice_v1/derivation_manifest.json) binds canonical episode hashes, source hashes/span IDs, query observation cutoffs and parent-query lineage. Canonical packets retain subject/branch/purpose metadata; verification receipts bind the exact episode and generator/solver implementation hashes. The main manifest also pins the episode compiler and existing SFT contract source hashes. There are no absolute machine paths, generation timestamps or tokenizer-dependent values in release contents, so same-code/same-seed regeneration is byte-identical.

The generation label path uses its sampled world parameters. The verification path separately reads/parses the **archived raw source**, verifies source/byte-span/event/ownership bindings, validates the fixed question and cutoff, reconstructs available register state, and computes numeric/ask/unknown targets. It does not call the generator's label path or trust an expected answer inside `evaluator.oracle`.

Only verified train/dev cases pass through the existing `verify_episode` receipt and `export_sft` eligibility compiler. Validation also recomputes stored receipts, flattened exports and derivation hashes rather than trusting a file's `verified:true` flag. Current source-use revocation can still block export of an historically correct answer: truth verification is not consent or training authorization.

These checks detect corruption relative to a trusted release manifest and inconsistent source/target edits. The manifest is not signed, the synthetic generator is the declared source authority, and someone replacing an entire consistent release plus its trusted fingerprint is outside this integrity check. Host provenance/permission storage, global deletion and trained-influence removal remain separate responsibilities. Changing seed produces a new release; the current fixed v1 identities must not be merged with a different-seed release without a new identity/version namespace.

## Commands and checks

Create a **new** output directory; the generator refuses overwrites:

```bash
.venv/bin/python scripts/make_episode_slice.py --output /private/tmp/episode_slice_reproduction --seed 23
.venv/bin/python scripts/make_episode_slice.py --validate data/diagnostics/episode_slice_v1
.venv/bin/python -m pytest tests/test_episode_slice.py -q
```

The targeted CPU run completed **12 passing tests**. They check byte-identical regeneration, role-before-derivation counts, explicit audit boundaries, late observation/correction cutoffs, ask/unknown/tombstone behavior, oracle exclusion, current permission revocation, permission escalation, raw-source/span/event tampering, rehashed incorrect SFT targets and overwrite refusal. All 20 release targets passed the raw-source solver; no model training or generated-answer quality was measured.

This closes a small source-bound preparation step for [S1.2](../stages/02_Data_and_Evaluation.md). The 2,000-target release still needs more independent worlds/task families, reviewed solver coverage, balanced action/task/length distributions, real-tokenizer capacity accounting, stratified review and a frozen release/exposure plan. Public ingestion, teacher expansion, broader task evaluation, authentic Replica data and a complete learned loop remain unimplemented by this slice.

# Prompt-v2 selected-record readout hypothesis

**26 October 2026 · H1 evaluated · first, restart and one-shot heldout passed.**
This document is complete before the first native job. It declares one
hypothesis, its exact prompt contract, its exact gate denominators, the
role-sealed label order, the two fresh-process jobs, the resource stop policy
and the acceptance evidence. The implementation is
`src/replica_lab/memory_bank_readout_v2.py`, the protocol is
`configs/memory_bank_readout_v2.json`, and the CPU contracts are
`tests/test_memory_bank_readout_v2.py`.

The predeclaration was committed before native execution. The guarded
`first02` and fresh-process `restart02` jobs completed; role-sealed heldout was
opened exactly once after every development gate passed. Results are below.

## Evidence basis

`native03` produced exact selected-record values at 124/128 = 96.875% and
passed every literal gate, but the MiniCPM readout answered 19/152 = 12.5%
exactly. Three separable failures explain that readout:

1. Every exact answer came from the paraphrase cohort in query forms 1 and 3;
   forms 0 and 2 produced meta-commentary. The prompt presented a question and
   a record, not a response contract.
2. 140/152 rows hit the 8-token cap. Even correct copies truncated random
   `val-…` strings (`val-51a85d9a21` → `val-51a85d9a`).
3. Unknown and revoked rows produced refusals (`Apologies, I'm unable…`)
   instead of the literal status word; the status was context, not an answer
   format.

The selected-record layer is therefore not the observed failure. The failure
is a readout-contract failure layered on exact host selection.

## H1 — readout contract hypothesis

> The 12.5% dev exactness is caused by answer-format ambiguity and an 8-token
> completion budget, not by wrong record selection. A static, predeclared
> instruction contract that requires exactly one line containing either the
> shown record value or the shown status word, with `max_new_tokens=32`, will
> satisfy the existing exact, correction, unaffected, unknown and revoked
> thresholds on both first-process and genuinely new post-restart query forms,
> with zero control false values.

This is a **readout-contract** hypothesis. Even a pass is bounded to "the exact
selected-record value or status is read out under a fixed contract". It is not
a persistent-learning, retention-of-learned-memory, calibrated-confidence,
learned-selector or compressed-memory claim.

### Interventions

- Static response grammar, no few-shot exemplars.
- `max_new_tokens=32`.
- Explicit permitted status vocabulary
  `UNKNOWN`, `REVOKED`, `AMBIGUOUS`, `SUPERSEDED`.

### Excluded interventions

- Few-shot exemplars or dev/heldout exemplar sets.
- Structured decoding, logits processors or grammar-constrained decoding.
- Host-selected candidate answers or answer reranking.
- Retries, trailing-value extraction or post-hoc answer repair.
- Threshold, split, fixture or denominator tuning.
- Relabeling the four bank abstentions.

## Prompt v2 (exact)

Built from runtime inputs only: the query text and either one authorized
selected value or one runtime status. No label object or expected answer is an
input.

```text
Response contract: Output exactly one line and then stop.
Do not explain, reason, quote, use Markdown, or add a prefix.
Do not invent values.
If a record value is shown, output exactly that value.
If a record status is shown, output exactly that status word.
Permitted status words: UNKNOWN, REVOKED, AMBIGUOUS, SUPERSEDED.
Question: {query_text}
Record value: {selected_runtime_value}
Answer:
```

For an abstention the value line is replaced by
`Record status: {UNKNOWN|REVOKED|AMBIGUOUS|SUPERSEDED}`. Allocation remains
exact `(entity, attribute)` matching against the live view minus the declared
view-only revocation. Decisions are byte-identical to the v1 bank decisions.

Normalization stays `strip + collapse whitespace + casefold`, so an uppercase
status word compares equal to the lowercase evaluation target.

## Roles and fail-closed heldout sealing

`prepared04/labels.json` is a single evaluation-only document. It is projected
once into disjoint role files:

`data/memory_causal_fresh_20261026/prepared04_roles/`

| Role file | SHA256 |
|---|---|
| `labels_dev.json` | `bc90652586365ee0f3da39e997e773a22b5fa437e394760984c4f0c8bef87309` |
| `labels_heldout.json` | `7443ee457ee644eabbd9db7a7595d8b5d341e37925a76a913110021c45281252` |
| `labels_train.json` | `b37da1908d437e910b24374d2a94de042d85779ed8577c7b5bad649d98039c16` |
| `manifest.json` | `17c6f8dc01f7d4142fc6d6c2cad8cbfebfa1c658c35be2f95aec17dc81244bc3` |

Source labels pin: `2cb282284c223802bc691b512474a4e051074d5490af5fb1db5748f63f81049c`.
Sealing command:

```bash
.venv/bin/python scripts/seal_memory_bank_roles.py \
  --prepared data/memory_causal_fresh_20261026/prepared04 \
  --output data/memory_causal_fresh_20261026/prepared04_roles
```

The projection is preparation-time only: no model, tokenizer, native job or
decision. Each role file carries the source label hash, its role, episode IDs
and `evaluation_only=true`. The manifest records per-file digests and
`heldout_written=true`; the role sets are checked disjoint and their union is
the twelve prepared episodes.

The runner opens exactly one role file at a time:

1. `labels_dev.json` after dev decisions are frozen and the literal gate is
   about to be applied.
2. `labels_heldout.json` only after first-process, restart, revocation, state
   and restart-accounting gates pass, and only after heldout decisions and
   heldout generations have been written without labels.
3. `labels_train.json` last, descriptive literal metrics only.

A failure at any earlier gate writes no heldout decision, generation, metric or
label read. The heldout bytes are never opened, parsed or hashed-beyond-pin
after a failure.

## Exact development denominators

Mechanically computed from the frozen dev bank decisions and the dev role
labels; pinned in the protocol and rechecked before generation. Denominators
are not tuned.

| Gate | Denominator | Required |
|---|---:|---:|
| Literal fact value exact | 128 fact rows | ≥0.85 |
| Literal unknown selections | 128 fact rows | 0 |
| Literal revoked selections | 128 fact rows | 0 |
| First-process exact answers | 72 (64 fact + 8 unknown) | ≥0.85 |
| First-process correction | 8 corrected-state `p0*` fact | ≥0.60 |
| First-process unaffected | 20 corrected-state `p1/p2/p3` fact | ≥0.70 |
| First-process unknown | 8 | ≥0.90 |
| First-process control false value | 8 unknown rows | 0 |
| Restart exact answers | 72 (64 fact + 8 unknown) | ≥0.85 |
| Restart correction | 8 corrected-state `r0*` fact | ≥0.60 |
| Restart unaffected / retention | 20 corrected-state `r1/r2/r3` fact | ≥0.70 |
| Restart unknown | 8 | ≥0.90 |
| Restart control false value | 8 unknown rows | 0 |
| Revocation exact `revoked` | 8 | ≥0.90 |
| Revocation control false value | 8 | 0 |
| Initial-state exact answers | 64 fact rows | ≥0.85 |
| Corrected-state exact answers | 64 fact rows | ≥0.85 |
| Format: single-line | every generated row | 100% |
| Format: ended by EOS | every generated row | 100% |
| Format: cap hits at 32 tokens | every generated row | 0 |
| Restart query-text disjointness | all restart query texts | 100% |
| Old first-process query regeneration | restart job | 0 |

Decision inventory: 152 dev decisions, 24 dev states, 4 dev episodes
(`epi-e-2b8ee3328d`, `epi-e-d4c2ae066c`, `epi-e-a9e9888b67`,
`epi-e-31535a8beb`).

**Residual abstentions.** Four of the 128 dev fact rows are bank abstentions
(`r-distractor` in two episodes × two states, `runtime_status=revoked`). They
remain in the fact denominator as misses and are never relabeled. The fact
ceiling is 124/128 = 0.96875, so the 0.85 gates remain feasible but tight.

`control false value` counts a status row whose generated answer exactly equals
any record value present in that episode/state's live or negative label
entries. Refusals and other non-value text are not counted as false values; they
still fail the exact-answer gate.

## Execution order

Preconditions, before any label or model access:

- verify `episodes.json`, `revocations.json`, `labels.json` and `manifest.json`
  hashes and the prepared manifest flags (`native_computation=false`,
  `model_loaded=false`, `tokenizer_loaded=false`);
- verify the read-only native02 report/checkpoint/ledger and the checkpoint
  bindings, counters and cached episode state;
- for restart, additionally verify the restart02 report/checkpoint/ledger, its
  loaded first-process checkpoint hash, `old_observations_reencoded=0`,
  `old_queries_reencoded=0`, `restart_new_queries=216`,
  `fresh_process_parity=true` and `gates.mechanical_pass=true`.

Job `first`:

1. Build and freeze the dev decisions; write `decisions_dev.jsonl`.
2. Open the dev role; apply the literal gate.
3. Stop before model/tokenizer if the literal gate fails.
4. Load MiniCPM exactly once; generate `first_process` rows.
5. Write `generations_dev_first_process.jsonl` and evaluate the first-process
   phase and format gates.
6. Stop; restart and heldout remain unproduced.

Job `restart` (fresh process):

1. Re-verify prepared inputs and both checkpoints in this process.
2. Read and verify the first job's immutable receipt, raw decisions and raw
   generations; require its first-process gate passed. A missing or failing
   receipt stops before the model.
3. Freeze the same dev decisions, write them fresh, open only the dev role and
   reapply the literal gate.
4. Load MiniCPM exactly once; generate `restart` and `revocation` rows.
5. Evaluate the restart gate, the revocation gate, the combined
   initial/corrected state gates and restart-query accounting.
6. Only if all pass: build and freeze heldout decisions, generate heldout rows
   without labels, write them, then open the heldout role once and report
   post-freeze confirmation metrics without tuning or retry.
7. Then open the train role for descriptive literal metrics only.

The restart phase uses new query texts (forms 2 and 3); the revocation phase
reuses the first-process distractor text and is a control, not a "new query".
`old_first_process_queries_regenerated` is the count of `first_process` rows in
the restart job and must be 0; `restart_new_queries` must be positive.

## Checkpoint and resource policy

- Two native jobs, strictly one after the other, one MPS worker each.
- One model load and one tokenizer load per job; `model_loads=2` total.
- MPS only; `PYTORCH_ENABLE_MPS_FALLBACK=0`; `PYTORCH_MPS_FAST_MATH=0`;
  deterministic algorithms; offline `HF_HUB_OFFLINE`,
  `TRANSFORMERS_OFFLINE`, `HF_DATASETS_OFFLINE`; capacity guard.
- Guard limits: 50 GiB driver, swap growth ≤512 MiB, minimum free ≥15%,
  sampled; ≤1200 s wall per job; artifacts ≤64 MiB; prompt ≤256 tokens;
  ≤152 decisions per split.
- Zero gradients, optimizer updates, checkpoint writes, base/extension exports
  and downloads. Native02 and restart02 are read-only; any attempted write
  fails the run.
- The restart02 checkpoint is a procedural post-restart anchor that is
  validated read-only. It is **not** consumed by generation; the readout prompt
  never contains cached vectors.

## Stop conditions

| Condition | Effect |
|---|---|
| Prepared/role/checkpoint pin or hash mismatch | stop before model and labels |
| First job missing, changed or first-process gate failed | restart stops before model, no dev labels |
| Dev literal gate failed | no model, no tokenizer, no generation |
| First-process format/exact gate failed | stop; no restart attempt |
| Restart or revocation gate failed | stop; heldout role never opened |
| State or restart-accounting gate failed | stop; heldout role never opened |
| Heldout failure | preserved; no retry, no relabel, no train tuning |
| Resource cap or deterministic-protocol violation | failure report; raw failure preserved |

Every failure path writes a report (or `failure_report.json` from the runner)
without heldout metrics, without `decisions_heldout.jsonl`,
`generations_heldout.jsonl` or heldout label bytes read.

## Native commands

Guard entrypoints are parent-owned. The parent must add
`run_memory_bank_readout_v2.py` to `scripts/run_capacity_preflight.py`
`--entry-point` choices; this slice does not edit that file.

```bash
# Role sealing (already produced and pinned above)
.venv/bin/python scripts/seal_memory_bank_roles.py \
  --prepared data/memory_causal_fresh_20261026/prepared04 \
  --output data/memory_causal_fresh_20261026/prepared04_roles

# CPU contracts
.venv/bin/python -m pytest tests/test_memory_bank_readout_v2.py -q -m "not integration"

# 1. First process
REPLICA_CAPACITY_GUARD=1 PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
.venv/bin/python scripts/run_capacity_preflight.py \
  --entry-point run_memory_bank_readout_v2.py \
  --report runs/memory_bank_readout_20261026/guard_first.json -- \
  --job first --device mps --config configs/memory_bank_readout_v2.json \
  --prepared data/memory_causal_fresh_20261026/prepared04 \
  --roles data/memory_causal_fresh_20261026/prepared04_roles \
  --output runs/memory_bank_readout_20261026/first

# 2. Restart (only after first passes)
REPLICA_CAPACITY_GUARD=1 PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
.venv/bin/python scripts/run_capacity_preflight.py \
  --entry-point run_memory_bank_readout_v2.py \
  --report runs/memory_bank_readout_20261026/guard_restart.json -- \
  --job restart --device mps --config configs/memory_bank_readout_v2.json \
  --prepared data/memory_causal_fresh_20261026/prepared04 \
  --roles data/memory_causal_fresh_20261026/prepared04_roles \
  --first-output runs/memory_bank_readout_20261026/first \
  --output runs/memory_bank_readout_20261026/restart
```

## Acceptance evidence

`first/report.json` must contain:

- prepared hashes and manifest flags;
- native02 checkpoint verification with `written=false`;
- `decisions_dev.jsonl` and `generations_dev_first_process.jsonl` hashes;
- literal gate, first-process phase gate (exact, correction, unaffected,
  unknown, control, format), and the mechanically recomputed denominators;
- `heldout.status=not_run`, no heldout/train artifacts;
- role-label access record with only `dev` opened;
- one model load, zero gradients/updates/checkpoint writes, wall and artifact
  usage.

`restart/report.json` must contain:

- both checkpoint verifications with `written=false`;
- the first receipt hash, decision hash and first-process gate pass;
- restart, revocation, state and accounting gates;
- restart-query disjointness counts and
  `old_first_process_queries_regenerated=0`;
- `old_observations_encoded=0`, `restart_new_queries>0`;
- if and only if all dev gates pass: heldout decision/generation hashes,
  heldout metrics marked `enforced=false`, and train descriptive literal
  metrics;
- on any dev failure: `heldout.status=not_run`, no heldout/train artifacts and
  no heldout label read.

Each generation row carries `prompt`, `prompt_sha256`, `prompt_ids_sha256`,
`generated_ids`, `generated_text`, `normalized_answer`, `ended_by_eos`,
`cap_hit`, `single_line`, `runtime_status`, `runtime_contract_exact`,
`selected_value`, `selected_record_id` and no evaluation field. Scoring joins
only by frozen decision key after generation.

Independent replay: recompute every metric and raw hash, verify the gate
denominators and confirm the absence of heldout/train artifacts on any failure.

## Implementation map

| Public API | Role |
|---|---|
| `validate_config` | freeze prompt v2, 32-token cap, thresholds and exact denominators |
| `build_role_projection`, `seal_role_labels` | role projection with per-file SHA256 |
| `load_role_labels` | pinned, single-role label open (`_read_role_json` is the test seam) |
| `verify_role_manifest` | pinned manifest, disjoint role sets and sealed-file presence without reading labels |
| `build_readout_split_decisions` | reuse the bank decisions and runtime values |
| `build_prompt_v2` | static one-line contract from runtime inputs only |
| `generation_row_v2`, `generate_row_v2` | label-free v2 row; single-line/cap/EOS flags |
| `build_model_generator_v2` | one sealed MiniCPM load; greedy, no logits processor |
| `decision_counts`, `validate_decision_counts` | mechanically pinned denominators |
| `score_phase`, `evaluate_phase_gate` | phase-scoped gates and format gates |
| `evaluate_state_gates` | combined initial/corrected 64-row gates |
| `restart_query_accounting`, `evaluate_restart_accounting` | new-query disjointness and old-query accounting |
| `verify_baselines` | read-only native02/restart02 checkpoint verification |
| `run_readout_job` | fresh-process `first`/`restart` orchestration with fail-closed ordering |

Reused unchanged: `build_split_decisions`, `annotate_decision_values`,
`freeze_decisions`, `write_frozen_decisions`, `query_text_register`,
`decision_status`, `normalize_answer`, `score_literal_values`,
`evaluate_dev_literal_gate`, `_labels_map`, `_target`,
`load_fresh_checkpoint`, `_unpack_cache`.

## Native results

| Evidence | Correct / rows | Rate | Gate |
|---|---:|---:|---|
| First-process exact answers | 72/72 | 100% | Pass |
| First correction | 8/8 | 100% | Pass |
| First unaffected | 20/20 | 100% | Pass |
| First unknown | 8/8 | 100% | Pass |
| Restart exact answers | 68/72 | 94.44% | Pass |
| Restart correction | 8/8 | 100% | Pass |
| Restart unaffected / retention | 20/20 | 100% | Pass |
| Restart unknown | 8/8 | 100% | Pass |
| Revocation exact `revoked` | 8/8 | 100% | Pass |
| Initial-state exact answers | 62/64 | 96.875% | Pass |
| Corrected-state exact answers | 62/64 | 96.875% | Pass |
| Heldout, one shot | **148/152** | **97.37%** | Post-freeze confirmation |

All generated rows are single-line and end at EOS with zero 32-token cap hits
and zero control false values. Restart texts are 100% disjoint from first-
process texts; old observations and old first-process queries are not
regenerated. Heldout was opened only after heldout decisions and generations
were written without labels, and was not retried or tuned.

The four heldout misses are the predeclared bank abstentions in two episodes ×
two states. The runtime answer is `revoked`, matching runtime status; the
scorer keeps those fact rows as misses rather than relabeling them. No false
value was generated.

## Resources and independent verification

The first guard took 70.94 s and the restart guard 205.86 s. Each loaded the
sealed model once. Sampled peak driver memory was 10,763,255,808 bytes, with
zero swap growth and no CPU fallback, gradients, optimizer updates, downloads,
checkpoint writes or extension exports. Artifacts are 424,576 bytes for first
and 1,217,677 bytes for restart.

The focused suite passes **21 tests**. The deterministic
[H1 replay receipt](../../reports/memory_bank_readout_20261026/replay_receipt_v1.json)
passes 26 checks: it rebuilds all 456 decision identities and replays **152 dev
plus 152 heldout generation rows**, every development phase/state/accounting
gate, raw hashes, implementation pins and zero-update/resource claims. Heldout
labels were not reopened or re-scored; only heldout structure, counts and
reported arithmetic were checked. `first02` and `restart02` bind the same
implementation hashes. The initial `first` failure was only a baseline source-
shape mismatch; its preserved receipt remains separate from `first02`.

## Status and non-claims

- **Implemented and measured:** H1 response grammar, role sealing, first and
  fresh-process restart execution, exact/correction/unaffected/unknown/revoked
  gates, and one-shot heldout confirmation.
- **Passed:** the predeclared useful post-restart readout endpoint on this
  synthetic fixture.
- **Not claimed:** persistent learning, learned-selector quality, compressed
  memory, calibrated confidence, general personal-memory competence or broad
  generalization. This is an exact selected-record readout contract only.

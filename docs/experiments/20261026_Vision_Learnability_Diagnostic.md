# Vision Learnability Diagnostic (predeclared protocol)

Status: **predeclared, not yet run.** This page records the two protocols and
their exact decision rules before execution. The parent appends results after
the scheduled jobs; no measured value belongs in this predeclaration.

Code:

- `src/replica_lab/vision_learnability_diagnostic.py` — model-free contracts.
- `scripts/run_vision_learnability_diagnostic.py` — native `--mode ablation`
  and `--mode gradient` runner.
- `configs/vision_mean_only_ablation_v1.json`
- `configs/vision_gradient_learnability_v1.json`
- `tests/test_vision_learnability_diagnostic.py` — CPU-only contracts.

## Question

Does the pinned train04 vision projector carry first-order image information in
its projected spatial residual, or only in the per-image mean projected patch?
Only if the mean-only ablation observes a spatial contribution is a bounded
gradient-inclusive learnability check allowed. Neither job reads development,
held-out or semantic-audit payloads, and neither runs V0.

## Shared inputs and pins

| Item | Value |
|---|---|
| Prepared root | `data/vision_balanced_alignment_v2_20261026/prepared04` |
| Protocol SHA-256 | `3bf636f00dce4e73168eb5f4b2c2454560276b32dfb1122b91d58ffcd44c2d77` |
| Manifest SHA-256 | `12f43b081e147a0c678e2137a321b709eb528099411b46b913735198563e0227` |
| Training alignment SHA-256 | `be6d62e32e688b7e4fc14ff327ec41df7062747e13d4c4d041cd7d05caa00bd8` |
| Training rows | 26 (14 answer + 12 caption) |
| Batches SHA-256 | `08484a4dbc000488c94c54510fad655c06251598a87294837e24d80cad09433a` |
| Train caption labels SHA-256 | `84788b08bd6c3a66f9c241e18db73ca8828c8dd15b62aad83e8d3bdf235b1954` |
| MiniCPM5 source | `checkpoints/minicpm5_source`, revision `f97400052a43d642bbc6e9975e2397e3ae6a6b52` |
| Model lock SHA-256 | `26dea74b4c543bb000c64ac7509171a399beee611ca0f7b70ebd941da64af792` |
| SigLIP2 source manifest SHA-256 | `8b2351f7b29f2bd7d881405dd978ae714a26c0170b7213d133a9e0059a199024` |
| train04 extension | `runs/vision_balanced_alignment_v2_20261026/train04/alignment/extension` |
| Extension weights SHA-256 | `76b8f7defcabb113e7c96f2eb9f5a4be8b8ed3886cd147264bb51cf013df2e3e` |
| Extension manifest SHA-256 | `593075106dfa6b91c5c0f27096861e1a57e0afc77b7838388e848926d666756a` |
| Extension updates consumed | 24, `base_saved=false` |

Stop set `{1, 130073}` (`</s>` and `<|im_end|>`), target EOS `1`, response-only
labels, one-position label shift and `mean_response_nll` are unchanged from the
prepared protocol. The declared matched controls are `correct`, `sibling`,
`shuffled`, `noimage`, selected deterministically by sorted pair id without
consulting target text.

Implementation pins reuse the prepared protocol pins with explicit read-only
migrations for the two previously repaired runners and for unrelated concurrent
user changes to `architecture/contracts.py`, `architecture/minicpm5.py` and
`memory_system/session.py`. Every migration is enforced at its declared current
hash; the two new learnability files are pinned explicitly. Any further byte
change fails before a model load.

## Mode 1 — mean-only projected-patch ablation

Predeclared artifacts:

- config `configs/vision_mean_only_ablation_v1.json`
- output `runs/vision_mean_only_ablation_v1_20261026`
- report root `reports/vision_mean_only_ablation_v1_20261026`

### Hypothesis

If train04's first-order image sensitivity is carried by the per-image mean
projected patch, replacing every valid projected patch with that image's mean
preserves the mean, removes within-image spatial variance and leaves the
correct/sibling/shuffled opposite-token gaps and greedy outputs largely
unchanged. Large changes indicate residual spatial information beyond the mean.

### Design

One model load and one train04 extension load, with two paired stages:

1. `train04_full_patch` — unmodified `adapter.project_visual`.
2. `train04_mean_only` — the same adapter and extension with an instance-level
   `project_visual` wrapper that, over valid patches only, replaces every
   projected patch by `mean(valid projected patches)`. Invalid positions stay
   zero. The image's mean is preserved; there is no global dataset mean.

The wrapper is installed only for the mean stage and restored immediately after;
the immutable `VisualInput` is checked before and after every projection. Full
stage projected statistics are computed before installation. The wrapper applies
to teacher-forced forwards, greedy generation and projected statistics.

Zero updates, zero optimizer steps, zero backward passes, zero gradient
computations, zero checkpoint writes and zero extension writes are required.
One model load, one extension load, no downloads, no CPU fallback, one exclusive
MPS worker, the capacity guard and the profiler are mandatory.

### Ablation invariants (all required)

- 26 rows, 104 records, 98 available records, 6 sibling placeholders, 98
  teacher-forced cells, 72 comparison cells, 98 generations.
- Fit coverage 26/26 with 14 answer and 12 caption rows and every row valid.
- For every image: 196 valid projected patches, maximum deviation from the mean
  `<= 1e-6`, within-image spatial variance `<= 1e-6`, mean preserved within
  `1e-6`, invalid positions zero, finite output, unchanged shape/dtype/device.
- Full and mean stages agree within `1e-6` on per-image mean patches, PCA
  content share, sibling and cross-pair mean cosine distances and
  `projected_mean_vector_rms_mean`; text embeddings and fit rows are identical.
- `projector_state_sha256`, the extension tensors and the sealed original
  parameter stamp are unchanged; no gradient exists on any parameter.
- Every available generation is nonempty with exactly one greedy hypothesis,
  answer `<= 8` and caption `<= 32` generated tokens, the declared stop set, and
  prefill argmax equal to the first generated token (98/98).
- Access contains only training rows and their referenced images.

### Baseline identity

`train04_full_patch` must reproduce attempt02's `train04_extension` on the same
26 prepared04 rows and controls: record/control counts, fit coverage, generated
ids and text, first-token gold/opposite/argmax tokens, gold rank and loss-token
counts exactly; first-token logits and opposite gaps within `1e-4`; mean NLL
within `1e-5`. Attempt02 did not retain full logits, so no full-logit
reconstruction is claimed. Any mismatch sets `baseline_mismatch`, forbids
ablation interpretation and stops without a rerun.

### Paired metrics and decision

For available rows, generated ids/text are compared between stages:

- `changed_sibling_rows` (20 available, maximum 20)
- `changed_shuffled_rows` (26, maximum 26)
- `changed_pair_rows = changed_sibling_rows + changed_shuffled_rows`
- `answer_gap_delta_row = gap_mean_only - gap_full` on the correct-image answer
  control (10 rows with a sibling), where `gap = gold_logit - opposite_logit`
- `answer_gap_reduction_rows`: `|gap_mean_only| <= |gap_full| - 0.10`
- `answer_gap_sign_flip_rows`

Per-kind and per-control summaries are reported; no aggregate-only acceptance.

Decision, with precedence:

1. `baseline_mismatch` if baseline identity fails.
2. `spatial_contribution_observed` if `changed_pair_rows >= 5` **or**
   `answer_gap_reduction_rows >= 3`.
3. Otherwise `spatial_residual_not_established`.

`observed` authorizes only the separate gradient diagnostic, never V0.
`not_established` stops the vision sequence. No automatic retry: a second
attempt requires a new predeclared config and fresh output.

## Mode 2 — gradient-inclusive learnability

Predeclared artifacts:

- config `configs/vision_gradient_learnability_v1.json`
- output `runs/vision_gradient_learnability_v1_20261026`
- report root `reports/vision_gradient_learnability_v1_20261026`

### Precondition

The gradient job refuses to load a model unless
`runs/vision_mean_only_ablation_v1_20261026/report.json` records
`spatial_contribution_observed`, `baseline_match=true`, satisfied ablation
invariants, zero gradients/optimizer steps and zero sealed access. The pinned
ablation config hash must also match.

### Schedule

The first seven updates of the pinned prepared04 batch file, unchanged:

- 7 updates × 4 slots = 28 row slots;
- exactly 2 answer + 2 caption rows per update;
- 26 unique rows (14 answer, 12 caption) by update 7, with two repeated caption
  rows (`vba2-w014-p0-a`, `vba2-w014-p0-b`);
- update 6 does not cover all rows; update 7 is the smallest full-coverage
  prefix. No new batch construction is allowed.

### Initialization and optimizer

Fresh seed-29 projector (`init_seed=29`, `fresh_seeded_normal_std0.02`), no
train04 warm start, frozen backbone/encoder, exactly six projector tensors and
11,542,016 parameters. Optimizer is `NativeFusedAdamW`
(`torch.optim.AdamW(fused=True)`), lr `3e-4`, weight decay `0.01`, betas
`0.9/0.999`, eps `1e-8`, max grad norm `1.0`, grad accumulation `1`.

### Backward and update evidence

Two backward-only probes before updates (one answer row, one caption row), each
with 6/6 projector gradients present, finite and nonzero, no non-projector
gradient, no optimizer step and no weight change.

For every update, before the step: finite `requires_grad` loss; original
gradients empty; the six projector gradients present, nonzero and finite;
finite gradient norm; unchanged sealed base stamp. After the step: positive
projector update norm, only projector tensors changed, unchanged sealed base
stamp. Across seven updates every projector tensor must change at least once.

Caps: 7 updates, 7 optimizer steps, 28 update backward calls, 2 probe backward
calls, 30 total backward calls, 900 s, 64 MiB, one model load, zero extension
loads, zero extension/checkpoint writes, zero downloads, no CPU fallback.

### Decoding and public controls

Exactly one greedy hypothesis per `(row, control)`; no sampling, beam, candidate
list or retry. Answer `max_new_tokens=8`, caption `max_new_tokens=32`, declared
stop set `{1, 130073}`, and 98/98 prefill argmax agreement. Denominators are
14 answer rows (10 sibling available), 12 caption rows (10 sibling available)
and 26 shuffled/no-image rows. Caption teacher-forced loss always uses the
fixed original caption target; following metrics use the supplied image's
sealed descriptors. Two caption shuffled donors are answer rows without caption
labels, so `caption_shuffled_following_accuracy` is reported over the 10
available caption donors and that denominator is stated; unavailable following
evidence is never fabricated.

### Exact train-fit gates (all must pass)

Answer:

| Gate | Threshold |
|---|---:|
| `answer_correct_accuracy` | ≥ 0.50 |
| `answer_correct_margin_sibling` | ≥ 0.10 |
| `answer_correct_margin_shuffled` | ≥ 0.10 |
| `answer_correct_margin_noimage` | ≥ 0.15 |
| `answer_sibling_following_accuracy` | ≥ 0.50 |
| `answer_prefill_gold_accuracy` | ≥ 0.50 |
| `answer_correct_vs_sibling_identical_rate` | ≤ 0.50 |
| `answer_correct_vs_shuffled_identical_rate` | ≤ 0.50 |
| `answer_correct_unique_texts` | ≥ 3 |

Caption:

| Gate | Threshold |
|---|---:|
| `caption_token_f1` | ≥ 0.20 |
| `caption_semantic_f1` | ≥ 0.50 |
| `caption_following_accuracy` | ≥ 0.50 |
| `caption_sibling_following_accuracy` | ≥ 0.50 |
| `caption_sibling_following_semantic_f1` | ≥ 0.50 |
| `caption_sibling_original_semantic_drop` | ≥ 0.15 |
| `caption_shuffled_following_accuracy` | ≥ 0.50 |
| `caption_shuffled_original_semantic_drop` | ≥ 0.15 |
| `caption_noimage_loss_margin` | ≥ 0.10 |
| `caption_noimage_semantic_f1` | ≤ 0.25 |
| `caption_dependence_f1_margin` | ≥ 0.00 |
| `caption_correct_vs_sibling_identical_rate` | ≤ 0.50 |
| `caption_correct_unique_texts` | ≥ 3 |

`caption_dependence_f1_margin` is the mean across sibling and shuffled controls
of (mean correct-image original-semantic F1 minus mean control-image original-
semantic F1). Answer and caption gate sets are independent; neither can offset
the other and failures are reported verbatim.

### Access and disposition

Only `training/alignment.jsonl`, `training/batches.jsonl`,
`training/train_caption_labels.jsonl` and referenced `procedural/assets/*` are
read. `evaluation/**`, development, held-out and semantic-audit roles are never
read; the runner never calls the development or held-out loaders. The access
order advances only `train -> train_fit`, and `train_fit` is entered only after
all seven updates.

- Engineering failure → `stopped_engineering_failure`; sealed roles stay closed.
- Any declared gate failure → `stopped_before_dev_heldout_v0`.
- All pass → `learnability_pass` and `v0_planning_authorized=true`, but V0 is
  still not run. V0 requires a separate predeclared job and the held-out role
  remains sealed.

Failure is terminal. No retry, no role-pure control and no alternate seed run in
the same job; a role-pure arm would need its own config and output and cannot
run simultaneously.

## Artifacts

Ablation output: `config.json`, `protocol.json`, `rows.jsonl`, `cells.jsonl`,
`fit_contract_rows.jsonl`, `fit_contract_coverage.json`, `features.json`,
`paired_ablation.json`, `report.json`, `manifest.json`, ledger and profile
artifacts.

Gradient output: `config.json`, `protocol.json`, `training_trace.jsonl`,
`backward_probes.json`, `fit_contract_coverage.json`,
`train_answer_generations.jsonl`, `train_caption_generations.jsonl`,
`scores/*.json`, `gates/train_fit.json`, `report.json`, `manifest.json`, ledger
and profile artifacts.

Fresh output paths are mandatory. Failures write `failure_report.json` and
preserve raw partial rows and traces.

## Implemented / measured / rejected / non-claims

Implemented before execution:

- strict mode-specific config validation, prepared/source/extension/baseline
  pins and implementation migrations;
- paired mean-only wrapper with per-image invariant receipts and restoration;
- baseline identity comparison at declared tolerances;
- paired swap/gap metrics and the precedence decision;
- seven-update prefix coverage audit and per-update role-mix audit;
- capped gradient trace, per-tensor ownership/finiteness and backward-probe
  validators;
- single-hypothesis generation evidence, training-only access audit and exact
  per-role gates;
- CPU-only tests covering these contracts.

Measured: nothing in this predeclaration. Results must be appended by the parent
after the jobs run in order (ablation first; gradient only if authorized).

Rejected: gradients or optimizer use in the ablation; a global-mean variant; a
role-pure or alternate-seed gradient arm; reading any sealed split; V0
execution; automatic retry.

Non-claims: these protocols do not promote learned quality, perception or
semantic competence; they are bounded training-only diagnostics over one small
procedural training set and one pinned checkpoint lineage.

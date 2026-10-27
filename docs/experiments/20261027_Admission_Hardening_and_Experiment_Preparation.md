# Independent fit admission and experiment preparation — 27 October 2026

## Scope and authority

Starting HEAD: `1304ab5d`. MiniCPM5 remains the baseline and
NativeFusedAdamW/`adamw_fused` the default optimizer. Historical Muon and custom
eager AdamW experiments retain separate provenance. The 50 GiB driver, 512 MiB
sustained swap-growth, 15% host-free, exclusive-worker and disabled-fallback
policies are unchanged. The protected runtime pilot was not read, imported,
modified or staged. **No native experiment, model download, development/held-out
evaluation, RSI cycle or personality adaptation was launched.**

## 1. Fit admission independently verifies the result

`scripts/run_nested_text.py` now recomputes fit correctness from authoritative
fit targets and preserved generation evidence. Saved `correct`,
`all_declared_gates_passed` and aggregate accuracy are comparisons, not trusted
inputs. `build_fit_receipt()` and `validate_fit_receipt()` enforce:

- scorer, answer-grammar and stopping-version bindings;
- decoded output, generated/content ID prefixes, token budgets, stop tokens and
  stop dispositions, sentence completion and truncation consistency;
- independent rescoring under the fit-only candidate vocabulary;
- configured minimum accuracy, exact attempt count and complete unique case IDs;
- explicit generation failures and complete source-free request audits;
- gate aggregate accuracy/failure counts and upstream flags only when they agree
  with the recomputed result;
- existing P-state, teaching run/cost, protocol, checkpoint and provenance checks.

Regression fixtures reject 1/28 correct with a 50% minimum despite an upstream
passing flag, incorrect text marked correct, stale/missing grammar metadata,
inconsistent generation/stop evidence, incomplete/duplicated cases and resealed
bad evidence. **255 admission/scoring tests passed.**

The reviewed current continuation receipt now pins runner SHA-256
`4e202fdb956ec1196311c9002b3ad864e7415dd84080d5c69558fe3c27062d88` and receipt
digest `83d58b41ef62f1fd8e47186afbf8f4489c051e900ef67d3acb29d743584f2213`.
Historical source blobs and old receipts are unchanged.

## 2. Development-label boundary

`RoleAccessGuard` denies every non-fit label role by default inside `load_rows`.
Only a fully validated passing fit receipt, or an explicitly non-training
protocol, opens later role parsing. `fit_candidate_answers_for()` reads only
eligible teaching/correction targets for the primary world/scope. Development,
unaffected, unknown/revoked, retention, wrong-world query and sealed answers
cannot enter fit scoring. `development_candidate_answers_for()` remains a
separate post-admission scope.

Tests prove label parsing fails before receipt validation and before native
loading. They also prove development labels are unavailable to the fit vocabulary
and that non-training receipt-free behavior remains intact.

## 3. Historical verification and preserved rejection

| Check | Result |
|---|---|
| Admission/scoring | **255 passed** |
| Source provenance and migrations | **174 passed** |
| Continuation historical replay | `exact_match`, zero failures |
| Historical vision accounting | `replay_matches`, **7,695 / 7,755** |
| Corrected step-one verdict | original `stopped_engineering_failure`; derived `step1_reproduction_supported` |

These replays do not reopen development or held-out labels and do not change the
historical verdicts. The response-CE fit remains rejected at 0/28, memory v4
remains rejected at 21/384 accepted-correct, and the vision curve remains
`curve_no_optimization_progress`. Failed native attempts, original validator
failures, the rejected caption decision and the eight-row exposure incident are
all preserved.

## 4. Memory experiment: value readout and abstention

`configs/value_readout_abstention_protocol_v1.json` and
`data/value_readout_abstention_20261027/fixture01/` provide a deterministic
chronological protocol, not a threshold sweep or writer-only arm.

- Seven episodes / 68 events: observation → future query → correction → unrelated
  retention query → unknown/wrong-world cases.
- 47 query cases, including 15 train, 12 disjoint training replay, 10 development
  and 10 sealed held-out cases. Training semantics, worlds and evaluation are
  disjoint; labels are rebuilt from the timeline and sealed by role.
- Four readout losses: future answer, correction, unknown abstention and
  wrong-world abstention. Only selected-expert `gate/up/down` receive them.
  Earlier-write gradients are **none**. Decoder, read projections, writers,
  backbone, router, personality and fast state remain fixed. Retention evaluation
  has no teaching gradient.
- One clock step per event and at most one optimizer step per teaching event.
  NativeFusedAdamW remains fixed. No second clock or budget02 revival.
- Seven distinct controls: learned value readout, exact context, selected-record
  literal readout, disabled, reset, shuffled values and wrong-world values.
  Exact context is a quality/cost reference, not a required winner.
- Gates require future-answer utility, correction, unaffected retention and
  abstention; local associative-loss improvement is not a gate.

Three independently trained arms each have 24 one-step teaching cases: the
treatment plus shuffled and wrong-world control fits. Thus the experiment ceiling
is 72 optimizer steps/backward calls, while only the 24-step treatment can bind a
fit receipt. The other 48 steps cannot admit development. Forward calls are 221,
writes 105 and checkpoints 3, all derived and validator-checked.

Protocol SHA-256:
`49b8bc451fe78f29e79ddbf5af2ea1a787739e98eedd9ef504025dfc758d5e32`.
Prepared manifest: `e544d41b62e9f28d4810a352d7a5e9ca20006c5a144325ffa7b15202ed87ee11`.
Validator: `c5d32f73314958837942b2417ce2d8c86d0cfb1a2e0b8927920cc550692d28b4`;
**10 tests passed**. No native admission runner or authorization exists yet.

Validate without model execution:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  scripts/validate_value_readout_abstention_v1.py \
  --config configs/value_readout_abstention_protocol_v1.json \
  --prepared data/value_readout_abstention_20261027/fixture01
```

## 5. MiniCPM5 smaller-step hypothesis

`configs/vision_smaller_step_lr3e5_v1.json` tests one variable: **lr 3e-5 versus
the completed 3e-4**, with the same MiniCPM5 parent, seed 29, four-example batch,
six projector tensors, response masks and NativeFusedAdamW.

The completed comparison is pinned at update-batch mean NLL
`10.964820146560669 → 12.560267925262451` (+1.5954477787017822), with generated
answers 0/14 and caption semantic F1 zero. The new schedule is prefix-0 cohort,
pre-update batch loss, one four-backward/one-step update, post-update batch loss,
then prefix-1 cohort. Update-batch loss is separate from the wider cohort.

Budgets before execution: 4,824 decoder forwards, 596 prepares, 400 explicit
logits, 196 generation prefill calls, at most 3,604 decode calls / 3,800 tokens,
168 visual observations/encoder-call ceilings, 4 backward calls, 1 optimizer
step, 1,200 seconds and 33,554,432 artifact bytes. One model load, no downloads,
extensions, checkpoints or retries. The run path forces offline mode and disabled
fallback, and includes the final report in the artifact cap.

The optimization gate allows at most +0.25 nats on the update batch and wider
answer/caption means. The independent functional fit gate requires generated
answer and prefill-gold correctness, caption improvement, correct-over-control
margins and bounded EOS/whitespace behavior. Loss/rank movement alone is not
conversion. Even a functional pass only permits a new predeclared protocol;
development, held-out, V0, promotion and retry remain closed.

Protocol SHA-256:
`931851ecc954cea91d97aa7a087c1a2bef528172379b06d453e6bec93ef4abad`;
**38 tests passed**. A real `run` path exists but has not run on MPS. Native
outcomes, generation lengths, wall time and artifact use remain unverified.

Review-only command:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONDONTWRITEBYTECODE=1 \
.venv-vision/bin/python scripts/vision_smaller_step_v1.py run \
  --config configs/vision_smaller_step_lr3e5_v1.json \
  --output runs/vision_smaller_step_lr3e5_v1_20261027 \
  --model-path checkpoints/minicpm5_source \
  --model-lock configs/minicpm5_sealed_lock.local.json \
  --authorization reports/vision_smaller_step_lr3e5_v1_20261027/authorization.json
```

**Do not execute this command from the current handoff.** It first requires a
committed protocol and a separate receipt bound to those exact bytes. Missing or
invalid authorization refuses before model loading.

## 6. Clean data lineage and bounded Spark preparation

The corrected caption lineage includes only the nine semantically matched
training candidates. The original parent decision remains `reject`; one mismatch,
five unclear, seven validation-only and all eight exposed rows are in the
21-row exclusion ledger. No exposed row is relabeled as untouched evaluation.

- Lineage manifest: `b67dd414d07a55ba2be386627860b1675197b46d40619013565cbb8241f8e691`.
- Rows: `b4efd15ad6b97e0df04f3d67233693d2772168e971c313ee2fce67ea554bc69c`.
- Exclusions: `07d9292350b562695a3cd1ce9416aaf8ca37613d3ee9227bdf224e4b52f0a817`.
- Admission defaults to `not_admitted` and requires a trusted local reviewed
  decision receipt bound to exact lineage/evaluation hashes and exactly the nine
  eligible child IDs. It is not a cryptographic signature and needs no secret.

A fresh evaluation lineage contains 24 new project-owned synthetic PNG scenes in
four families. Analytic geometry/color probes and authoritative scene specs
verify image-caption agreement without model or human label generation. The
partition is disjoint from all 30 external rows and sealed before any model use:

- Evaluation manifest: `843e5da40d9c190585375a363a4d97b6366a8c6c78969fbdedce6cd3d9da3ce1`.
- Seal: `f2179f3c02d30309362ed6c0913604d646327412dceb60600adc42e7b6f30a6e`.
- Image aggregate: `c1a824f5763d13a0e865bcda5f5b3dc46fe078d5eb30d133ec35c007766aedb3`.

All 38 lineage/Spark tests passed. Missing admission and missing Spark
qualification both refuse with exit 2. Spark official BF16, official FP8 storage,
dequantized derivatives and native FP8 execution remain separate. Analytical
estimates put extension BF16 residency at 11,594,609,442 bytes within declared
headroom, full-base BF16 at 87,256,869,666 bytes (blocked), and native FP8
training as unquantified. These are estimates, not measured fit. A later filtered
text-only self-replay retention ablation is documented but unauthorized.
Unqualified visual answers are never ground truth.

## 7. Remaining capability gaps and admission requirements

No passing production fit receipt, useful future-answer memory, correction,
unaffected retention, generated visual correctness, causal image dependence,
native Spark training qualification or neural RSI capability is established.
Publication, rollback and recovery prove infrastructure only.

Next concrete work is CPU-only: implement/review the memory admission runner,
then require its independently validated 24-step treatment fit receipt before
any development labels are read. The smaller-step vision runner additionally
requires its committed protocol and separate authorization before any native
launch. Neither protocol permits automatic retry, development, held-out or RSI.

## 8. Copy-ready continuation prompt

> Continue from the current HEAD. Inspect `git status`, intervening commits,
> `AGENT_START.md`, `execution_state.md` and
> `docs/experiments/20261027_Admission_Hardening_and_Experiment_Preparation.md`.
> Preserve concurrent work and all historical evidence. Do not read, import,
> modify or stage `src/replica_lab/runtime_pilot.py`. Keep MiniCPM5,
> NativeFusedAdamW and the 50 GiB / 512 MiB / 15% native guards unchanged. Do not
> launch native execution from this handoff.
>
> First rerun the 255 admission/scoring tests, 174 provenance tests, 10 memory
> protocol tests, 38 vision smaller-step tests and 38 data/Spark tests. Replay
> continuation into a fresh output path and run `replay-accounting --check` and
> `replay-step1 --check` against the preserved receipts. Do not overwrite old
> reports or change historical pins. Use the reviewed current continuation
> receipt for the hardened runner and keep immutable snapshots unchanged.
>
> Next implement only the CPU-side memory admission runner/review glue for
> `value_readout_abstention_v1`. Preserve one step per event, the three-arm
> 24+24+24 derivation and the 24-step treatment-only fit receipt. Do not read
> development/held-out labels until that receipt independently passes. Keep
> exact-context, selected-record, disabled/reset, shuffled and wrong-world
> controls separate. Do not require useful memory to beat exact context.
>
> The smaller-step vision `run` command is prepared for later review only. It
> requires a committed protocol and a separate receipt bound to config SHA-256
> `931851ecc954cea91d97aa7a087c1a2bef528172379b06d453e6bec93ef4abad`.
> Do not launch it without new authorization. A fit pass is required before any
> development protocol; loss/rank changes alone are not functional conversion.
> Keep the rejected caption tranche and eight exposed rows out of training and
> untouched evaluation. Use only the new nine-row reviewed lineage after explicit
> admission and keep the 24-scene synthetic evaluation sealed. Keep Spark backend
> work analytical/preparatory and neural RSI/personality adaptation deferred.

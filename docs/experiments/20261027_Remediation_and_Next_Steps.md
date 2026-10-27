# Review remediation and bounded step-1 result — 27 October 2026

## Scope and lineage

Starting review: `d24f8030`. MiniCPM5 remains the baseline and
NativeFusedAdamW the default optimizer. The 50 GiB native driver limit, 512 MiB
sustained swap-growth limit, 15% minimum host-free memory, exclusive native
worker and disabled CPU fallback are unchanged. The protected runtime pilot was
not read, imported, modified or staged.

Focused implementation and evidence commits:

| Commit | Content |
|---|---|
| `dfe39a07` | Admission, scorer, provenance, memory/QA/RSI work and predeclared step-1 protocol |
| `f38bab72` | Initial step-1 authorization |
| `d9c7393e` | Pillow dependency correction before any readout/update |
| `cf37d85` | Attempt-02 accounting-stop evidence and superseded accounting |
| `ca8669b9` | Raw-generation and update-visual-work protocol correction |
| `ac12ad0` | One corrected native launch authorization; no further retry |
| `3c9c7cf` | Original completed attempt-03 trajectory and validator failure |
| `3194d37` | Corrected validator and derived supported step-1 decision |

No Spark training, model download, development/held-out evaluation, RSI cycle,
extension save or duplicate base/checkpoint save occurred.

## 1. Evaluation admission and negative scoring

`scripts/run_nested_text.py` now writes and validates
`replica.nested_text.fit_receipt.v1`. Behavioral development evaluation requires
a passing receipt before `torch.load`, native worker/model loading or development
role access. The receipt binds:

- exact P-state file bytes and payload binding;
- protocol/config and episode manifest, role files and world identities;
- scorer, answer grammar, stopping policy, generation limits and candidate answers;
- required case IDs, complete coverage, attempts, source-free audit and gate report;
- actual teaching run, teaching cost, P-only/checkpoint hashes and resulting
  canonical state.

Missing, failed, stale, incomplete, cross-state, wrong-scorer and generation-
failure receipts fail closed. Historical teach artifacts lacking the new state
hashes cannot back a new passing receipt; this does not rewrite them.

The shared grammar now handles terminal punctuation, Markdown and declarative
responses while requiring a standalone expected phrase. Decimal/thousands and
possessive extensions, competing answers, ambiguity, contradiction, empty and
truncated generations are rejected. `unknown` and `revoked` remain exact-only.
Generation failures have an explicit gate and accurate failed-attempt counts.
New rows carry `replica.nested_text.answer_grammar.v2.1.1`; historical scorer
receipts are unchanged.

Focused result: **232 admission/scoring tests passed**. The separately versioned
`replica.nested_text.scoring_replay.v2.1` derived replay scored the same 196
preserved outputs with **no group-count changes** from the historical replay. It
is a derived measurement, not a new model evaluation.

## 2. Historical replay versus current qualification

The original provisioned failures were reproduced and preserved:

- four vision migration-pin failures in `baseline_vision_pins.xml`;
- continuation replay failure with four `run_capacity_preflight.py` drift errors
  in `baseline_continuation_replay.json`.

The remediation adds content-addressed immutable source snapshots and separate
reviewed current-source receipts. It does not change old config hashes or old
receipts, skip failing tests or accept temporary source caches.

| Source authority | Identity / result |
|---|---|
| Vision snapshot | `e384cb11d3a076ad073d938b4b0c76f4a6f3caf1bd1cbd2d3ab3e776fa305ff7`, 27 entries |
| Continuation snapshot | `8a3cd882a48ea29698b8a5e07077f066268af604ecd9d8b52c6203b50870a1ef`, 10 entries |
| Continuation historical replay | `status=exact_match`, `exact_status=passed`, zero failures |
| Replay/current tests | **148 passed**, including genuine tamper and guard-map/command rejection |
| Spark final-product bundle | 43 inputs, 3,733,780 bytes, digest `ceebdb951de3a771c0eecdeeae38c47875565e1fa87c2da9ef0eee055a309d4a` |
| Spark verifier | **436 / 436 checks passed**; 21 historical-bundle tests and 15 final-product tests passed |

The Spark verifier now links acceptance v4, calibration rejection, independent
16-expert continuation and the latest vision curve. It distinguishes reviewed
input changes from unexplained drift and works without temporary source-cache
locations. Official BF16, official FP8, dequantized derivatives and native FP8
execution remain separate cases; no full-training fit is inferred.

## 3. Correct compute accounting

The first reconstruction was itself corrected against raw saved IDs. The
superseded 5,254-token artifact is preserved under
`reports/remediation_20261027/superseded/`.

| Prefix | Saved generated IDs | Reconstructed scoring decoder passes |
|---:|---:|---:|
| 0 | 1,362 | 1,852 |
| 1 | 1,442 | 1,932 |
| 2 | 1,396 | 1,886 |
| 4 | 721 | 1,211 |
| 7 | 324 | 814 |
| **Total** | **5,245** | **7,695** |

The historical 1,470 count is exactly prepare/sense work, not total scoring
forwards. Adding 60 training/probe events gives **7,755** total events. These are
reconstructions from preserved records, not measurements by the old run.

New v2 instrumentation counts physical decoder forwards by phase, generated
prefill/decode work, text/visual positions, visual observations, encoder
calls/hits/misses, projector passes, backward calls and optimizer steps. Logical
scoring cells/operations remain separate. Per-readout and cumulative budgets use
explicit units and fail at call time. Final v2 result: **51 tests passed**;
accounting replay returns `replay_matches` at **7,695 / 7,755**.

## 4. Memory utility diagnosis

The derived diagnosis uses train/dev evidence only and preserves the historical
acceptance gates.

| View | Address candidate | Value candidate | Value oracle |
|---|---:|---:|---:|
| Accepted fact layers correct | **62 / 62** | 57 / 156 | 252 / 252 |
| Strict accepted-correct fact groups | **4 / 256** | 14 / 256 | 102 / 256 |
| Partial one-layer groups | 54 | 23 | 48 |
| Wrong accepted fact layers | **0** | 99 | 0 |
| Unknown false recall | 0 / 32 | 10 / 32 | 10 / 32 |
| Revoked false recall | 0 / 16 | 3 / 16 | 3 / 16 |

The **62/62 versus 4/256** result is an aggregation distinction: query acceptance
is `any(layer)`, while strict group correctness is `all(layers)`. The 54 partial
address groups contain one correct accepted layer and one abstaining layer, not a
wrong address. Address acceptance remains low: 198/256 fact groups accept no
layer.

The next bottleneck is **value readout plus abstention**, not address recovery or
another threshold sweep. Exact-context, selected-record literal readout,
disabled/reset memory, shuffled values and wrong-world controls remain separate.
The derived receipt is
`reports/remediation_20261027/memory_acceptance_diagnosis_20261027.json`;
**32 memory tests passed**.

Under the current one-step-per-event path, only the selected expert's `gate`,
`up` and `down` tensors receive teaching/correction gradients. Decoder/read
projections, writers, backbone, router, personality and fast state receive none.
Current unaffected development rows receive no teaching gradient. No new memory
architecture or training arm was added.

## 5. MiniCPM5 bounded vision hypothesis

### Historical verdict and audits

The v1 curve remains `curve_no_optimization_progress`: answer/caption NLL fell
65.40%/69.68%, prefix 1 overshot both losses, answer accuracy remained zero and
rank movement was not functional conversion. New audits separate EOS-only and
whitespace-only outputs, verify target equality and response masks, and retain
correct/sibling/shuffled/no-image controls.

### Attempts and preserved failures

1. Attempt 01 stopped before readouts/updates because the base environment lacked
   Pillow. The existing locked `.venv-vision` overlay supplied Pillow 12.0.0;
   no package was installed or downloaded.
2. Attempt 02 reproduced exact prefix-0/null parity and executed one update, then
   stopped at visual-encoder observation 217 because the 216 cap omitted the
   four update-phase observations. Its raw outputs and analysis are preserved.
3. Attempt 03 completed all three readouts and the update. Its original report is
   `stopped_engineering_failure` because final validation double-counted visual
   counters and rejected two valid cache hits. The original report is unchanged.

A CPU-only derived replay corrects that validator, not the raw measurements.
Its receipt is
`reports/remediation_20261027/step1_attempt03_derived_validator_correction.json`.

### Measured result

The derived decision is **`step1_reproduction_supported`**: the repeated prefix-0
null readout is exactly deterministic, and the first NativeFusedAdamW update
increases both training-only teacher-forced losses.

| Measurement | Prefix 0 / null | Prefix 1 after one update |
|---|---:|---:|
| Answer teacher-forced NLL | 10.704498972211566 | 11.74190446308681 |
| Caption teacher-forced NLL | 9.773420969645182 | 13.448152462641398 |
| Generated answer accuracy | 0 / 14 | 0 / 14 |
| Answer prefill-gold accuracy | 0 / 14 | 0 / 14 |
| Caption semantic F1 | 0 | 0 |
| Projector SHA-256 | `511ba5c7…` | `e6f1d25e…` |

The update had loss 10.964820146560669, gradient norm 345.9010314941406 and
projector update norm 1.0185893941275908. All six projector tensors received
finite nonzero gradients; original backbone gradients were absent.

This supports only the **first-step overshoot hypothesis**. Prefix-1
correct/sibling and correct/shuffled identical-output rates are 0.20 and
0.0714; caption following/semantic metrics remain zero. Generated-answer
correctness and causal image dependence are not established. Development,
held-out, V0, promotion and further retry remain closed.

### Actual work and resources

- Decoder forwards: **5,644** = 4,166 generation + 882 prepare/sense + 588
  explicit logits + 8 update forwards.
- Visual observations: **220**; encoder calls/misses: **218**; cache hits: **2**.
- Projector passes: **4,594**; backward calls: **4**; optimizer steps: **1**.
- Model loads: **1**; downloads, extension writes and checkpoint writes: **0**.
- Report wall time: **637.686 s**; console wall time: **640.08 s**.
- Guard peak driver: **11,932,663,808 bytes**; profile sampled peak:
  **11,932,680,192 bytes**. Swap growth: **0**; guard stop reason: `null`;
  host-free at end: **71%**.
- Final run artifacts: **1,498,052 bytes**, below the 16 MiB limit.

Replay without native execution:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  scripts/run_vision_learnability_curve_v2.py replay-step1 \
  --check reports/remediation_20261027/step1_attempt03_derived_validator_correction.json
```

Expected: `replay_matches`, original `stopped_engineering_failure`, corrected
`step1_reproduction_supported`.

## 6. Dataset asset QA

The local 30-image tranche passed 2,656 machine checks and decoded all images,
but admission is **rejected**: 9 training captions matched, 1 mismatched and 5
were unclear. Seven development rows remain validation-only. An initial template
also exposed eight sealed rows to semantic review; they are quarantined and no
longer valid as untouched evaluation for this workstream. The pre-correction
queue, contact sheets and incident receipt are preserved. See
[the asset-QA record](20261027_External_Caption_Asset_QA.md).

## 7. RSI remains conditional

The [conditional candidate plan](20261027_Conditional_RSI_Candidate_Plan.md)
freezes comparison policy, gradient scope, independent evaluation, rollback,
rejection examples and total-compute accounting. Existing candidate publication,
rollback, recovery and neural-deferral tests pass **19 / 19**. No neural RSI
candidate was published. Useful recall, correction, unaffected retention and
visual grounding must pass independently before `NeuralGate` can be reconsidered.

## 8. Final verification

The [final verification receipt](../../reports/final_remediation_20261027/summary.json)
binds the tested commit `f366a4f2d6da7ce26ff76437c9cd5fc0fd7eb575`.

| Scope | Result |
|---|---|
| Provisioned full suite | **2,677 passed**, 0 failed, 9 skipped, 21 subtests |
| Complementary Pillow overlay | **52 passed**, 1 intentional complementary skip |
| Fresh `--no-hardlinks` clone | **2,630 passed**, 0 failed, 56 explicit artifact/Pillow skips, 21 subtests |
| Tracked artifact guard | Passed on **11,009 files** in both trees; 74 guard tests passed |
| Clean-clone Spark verifier | **436 / 436 checks passed**, 43 inputs, no source-cache dependency |

The provisioned skips are seven Pillow JPEG tests and two official-PIL branches;
the overlay covers all nine. Clean-clone skips are explicit missing local
model/data/run artifacts and provisioning, recorded by name/message in its
JUnit. The initial two-failure full-suite receipt is preserved; reverting the
optional link in a pinned RSI document fixed the cause without changing old
hashes or skipping tests. The clean clone remained clean.

## 9. Next executable instruction prompt

> Continue from the current HEAD; inspect `git status`, intervening commits and
> `execution_state.md` before editing. Preserve concurrent work and do not read,
> import, modify or stage `src/replica_lab/runtime_pilot.py`. Keep MiniCPM5,
> NativeFusedAdamW and the existing native resource guards unchanged. Do not
> launch another native experiment or open development/held-out data from this
> handoff.
>
> First reproduce the CPU evidence: run the admission/scoring, replay/provenance,
> memory, Spark, asset-QA, RSI and v2 accounting tests; run
> `replay-accounting --check` and `replay-step1 --check` against the preserved
> receipts. Confirm the historical four vision failures are represented by
> explicit snapshot replay, not deleted tests. Use fresh output paths and keep
> original reports unchanged.
>
> The next learning design must target useful future answers, correction and
> unaffected retention under the existing one-step-per-event contract. Localize
> value-readout and abstention errors before changing architecture. Declare the
> exact selected-expert gradients and keep decoder/readers/writers/personality
> fixed unless a new reviewed protocol says otherwise. A valid passing fit receipt
> is required before any dependent development evaluation.
>
> For vision, treat `step1_reproduction_supported` as evidence of first-step
> overshoot only. Do not infer generated-answer correctness, causal image
> dependence or permission for development. If a new experiment is justified,
> predeclare one primary hypothesis addressing the measured overshoot, with
> correct/sibling/shuffled/no-image controls, physical-work budgets and a fresh
> protocol commit/authorization. Do not repeat the step-1 run or change budget02.
> Replace the exposed evaluation tranche with a fresh independently sealed one.
> Keep Spark backend/data preparation independent and keep neural RSI deferred.

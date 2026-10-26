# One-clock nested-text MiniCPM5 baseline

**26 October 2026 · native integration and source-free evaluation implemented;
development learned utility rejected; held-out evaluation unopened.** Sealed-
final rows were not parsed or scored. Integration is committed at `1648f72d`;
runner and review corrections are `ff5e4a9e`, `5a78db3`, `ec602a4` and
`6739804`. The protocol is `configs/nested_text_nl1_v1.json`, with synthetic,
role-separated episodes in `data/nested_text_nl1_20261026/`. The selected
development receipt is `p_only_dev04`; `p_only_dev03` remains preserved.

## Implemented and trained scope

The runner connects the sealed MiniCPM5 backbone to the standalone NL0 owners:
16 fixed-routed plastic experts, width 256 and hidden width 1024, FP32
`NativeFusedAdamW`, one selected expert per eligible event, one clock and no
replay. The objective is local selected-expert half-sum squared error,
$L=\frac12\sum_i(\hat y_i-y_i)^2$, on detached projected features, not response
cross-entropy or mean-normalized MSE. Earlier metadata's `mse` label was
imprecise; the computation is unchanged and both world protocol hashes remain
identical. Only the selected expert receives an update; backbone and read path
remain fixed.

Teaching exports both a full checkpoint and a weights-plus-bindings P-only
artifact. P-only evaluation rejects optimizer, moments, clock, dedup, revision,
coordinator, observation, source and KV state. Its requests contain no source
text or memory records and updates are disabled. Controls include preteaching,
frozen/reset P, shuffled routing, wrong-world P, wrong-world-on-primary and
spine-absent parity.

## Native execution

| Run | Measured scope | Result | Peak sampled driver | Swap growth |
|---|---|---|---:|---:|
| `teach02` | 24 teaching +3 correction events; 27 optimizer steps; 117 response tokens | Completed; full and P-only exports written | 11,815,763,968 B | 0 B |
| `wrong_world_teach` | 24 wrong-world events; 24 optimizer steps; 92 response tokens | Completed; independent world-beta state written | 11,815,763,968 B | 0 B |
| `p_only_dev04` | 196 development/control scoring attempts; 196/196 source-free audits; zero updates | Development gates failed; held-out evaluation closed | 11,815,763,968 B | 0 B |
| `resume01` | Fresh-process full-checkpoint load; next `correction-0003` update | Exact post-update state equality; revision 27→28; one transition | 10,756,947,968 B | 0 B |

Primary teaching took 10.14 s end-to-end and exported a 107,529,763-byte full
checkpoint plus a 50,347,007-byte P-only state. Wrong-world teaching exported a
101,206,264-byte full checkpoint and the same-size P-only format. Dev04 took
123.61 s and generated 1,566 tokens across all 196 attempts; the 60 primary
score requests account for 479 of those tokens. Resume took 8.07 s. Minimum
observed host free memory across these runs was 75%; guard stop reasons were
null and CPU fallback remained disabled.

The resume reference is a second independently loaded learner in the same
fresh process. Its after-state hash exactly equals the resumed learner after
the same next event. This validates checkpoint restoration and deterministic
update composition; it is not an independent uninterrupted-process comparison.

Dev04 persists every preteach role and audits every scoring attempt. Its scores,
controls and non-cost metrics are identical to dev03. The earlier run audited
104 attempts in aggregate, retained 152 distinct attempts and omitted 44
preteach attempts from serialization; those evidence gaps are corrected, not
rewritten. `p_only_dev` and `p_only_dev02` remain preserved as the SQLite-parent
and wrong-world-binding implementation failures.

Raw evidence is under `runs/nested_text_nl1_20261026/`: teaching transactions,
parameter audits, both exports, all development generations and controls,
access receipts, failure analyses, profile samples and guard host samples.
`p_only_dev04/analysis.json` records the selected failure and dev03 comparison.

## Development result

| Metric | Result | Declared gate |
|---|---:|---:|
| New-instance accuracy | 0/16 | ≥0.50 |
| Gain over preteaching | 0.00 | ≥0.10 |
| Correction accuracy | 0/4 | ≥0.60 |
| Unaffected accuracy | 0/8 | ≥0.70 |
| Legacy retention | 0/8 | decline ≤0.02 |
| Unknown false recall | 0/8 | ≤0.10 |
| Revoked false recall | 0/4 | ≤0.25 |
| Wrong-world own accuracy | 0/12 | control only |
| Teaching duty cycle | 0.999938 | ≤0.10 |
| Full checkpoint bytes | 107,529,763 B | ≤268,435,456 B |
| P-only bytes | 50,347,007 B | ≤134,217,728 B |
| Warm p95 overhead | 514.48 ms | ≤2,000 ms |

Preteaching, frozen, reset-P and shuffled-route accuracies are also zero.
Spine-absent parity passes, but there is no learned gain for any control to
explain. Raw answers are generic refusals or incomplete text rather than the
taught synthetic names. The 27 committed half-sum-squared-error transactions
establish that updates occurred; they do not establish useful language readout.

The failed gates are `min_new_instance_accuracy`, `min_gain_over_preteach`,
`min_unaffected_accuracy`, `min_correction_accuracy` and `max_duty_cycle`.
Checkpoint and P-only byte gates pass. `gates/access_final.json` records
`heldout_opened=false` and the development barrier failure. No sealed-final
rows were parsed or scored and no held-out claim exists. The episode-manifest
loader hashes `sealed_final.jsonl` for lineage before development, so this is
not a claim of zero sealed-file I/O.

## Verification and disposition

The focused CPU suites for nested text, nested learning, the architecture
adapter, memory session and fast ports pass **112 tests**. Exact-loss and
checkpoint-gate boundary assertions cover the review findings. Native teaching,
P-only loading, wrong-world controls and fresh-process resume exercise the
integrated paths on MPS with one exclusive worker and the unchanged
50 GiB / 512 MiB / 15% guard.

Independent static/JSON/hash review confirmed the negative terminal decision
and identified the imprecise objective label, missing checkpoint-byte gate and
incomplete audit/token scopes. Commit `6739804` corrects those defects; dev04
verifies them natively without changing the model result. Final independent
closeout is **correct** with no remaining findings. Review was static/read-only;
the reviewer did not run tests or native code.

**Implemented:** one-clock native integration, selected-expert teaching,
strict P-only export/load, complete source-free controls, access barriers and
checkpoint resume. **Trained:** only the declared plastic bank on teaching and
wrong-world events. **Rejected:** development learned utility and the duty-
cycle gate. **Unestablished:** new-instance gain, correction, retention,
multi-seed utility, additional clocks, learned routing and promotion.

No additional clock, objective, visual-memory arm, denser port or held-out retry
is authorized by this result. The next work must remain development-only and
first explain the existing readout's null generation behavior and teaching
cost; it must not tune against sealed-final rows or start held-out evaluation.

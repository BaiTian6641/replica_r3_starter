# M4 — static pattern and deterministic routing rejected

24 October 2026 UTC. **Decision: quality reject on general noninterference.**
The independent [verdict](../../reports/m4_pattern_attempt04/verdict.json) passed
all engineering checks except `general_noninterference`. It is not a resource,
lineage, restart or optimization defer.

Sections 1–4 preserve the original no-warmup M4 record. The later
static-optimization arm is recorded separately in section 5; it does not reverse
that verdict.

## 1. Delayed and general outcomes

| Metric | Result | Interpretation |
|---|---|---|
| Router dev / held-out / regression | 4/4 / 6/8 / 8/8 | Above null controls; no static-pattern attribution |
| Direct pattern dev / held-out / regression | 0/4 / 0/8 / 1/8 | Poor static utility |
| Pattern training accuracy | 8/24 | Underfit, final mean loss 1.13375 |
| General correct / router / fast | 0/4 / 0/4 / 0/4 | No retention repair |
| Adversarial vs disabled | 0/4 vs 2/4 | **Mandatory rejection** |
| Development adversarial | 0/2 vs disabled 2/2 | Also below base |
| Full context | 4/4 | Items are answerable |

All 20 delayed dev/held-out/regression router evaluations selected `fast`
because of uncovered hashes. Router and fast token IDs were identical, so router
performance is not evidence of static-pattern utility. The adversarial arm used
fast memory with `pattern=None`; therefore warming up pattern tables cannot
repair that specific gate under the current wiring.

## 2. Engineering evidence

- Causal 2/3-gram hashing, independent pattern branch, zero-init output and
  sparse gradient/optimizer behavior are implemented.
- Update-two gradients are finite and nonzero across all pattern tensors.
- Sparse gradients are densified only for AdamW; Muon rejects sparse gradients.
- Disabled-base token IDs match the parent and fast modules are unchanged.
- Same-process depth-image reload passes. Fresh-process restart was not run
  because the candidate was rejected.
- Table parameters occupy 134,217,728 bytes; optimizer state is 269,527,088
  bytes (1,056,768 int8 and 268,470,320 FP32).
- The preserved verdict lacks complete collision/latency fields from the failed
  runner. Do not infer those measurements; a separate collision evaluation is
  required before claiming them.

## 3. Costs and preserved failures

- 16 updates, 426 answers, 1,386 generated tokens and 28,241 prompt tokens.
- Profiled wall 154.291 s; sampled peak driver 11.127 GiB; no downloads or base
  copies. One extension and one depth image were saved.
- Attempt 1 failed on a generator-scope error; attempt 2 omitted session
  initialization; attempt 3 hit SparseMPS `ne` during gradient audit; attempt 4
  completed training/evaluation but failed verdict binding. All are preserved.
- Independent verdict with pinned parent adversarial controls: quality reject.

## 4. Next constraint

A future static-optimization diagnostic must directly measure pattern utility
on covered non-training cases and enable pattern in adversarial evaluation. A
source-LM warmup would otherwise be additional training with a predetermined
rejection and no routed-static-utility measurement. Do not reclassify this M4
failure as accepted, and do not advance to M5 on the strength of fast-memory
scores. Retention remains open.

## 5. Static-optimization arm — v7 current evidence

The final corrected arm used [protocol v7](../../configs/m4_pattern_static_v7.json)
(implementation `ad09a77`, protocol commit `d1364fc`). Raw [run](../../runs/m4_static_attempt07/run_report.json)
and independently scored [final verdict](../../reports/m4_static_attempt07/verdict_final.json)
(the [initial scorer artifact](../../reports/m4_static_attempt07/verdict.json) is preserved)
with the [guard](../../reports/m4_static_attempt07.guard.json) and
[collision audit](../../reports/m4_static_attempt07/collision_audit.json).
**Decision: reject; diagnostic only.** Sixteen of twenty total checks pass. All
listed engineering, correction and same-process restart checks pass; these four
capability/noninterference gates fail: `general_noninterference`,
`direct_pattern_covered_regression`, `direct_pattern_covered_dev`, and
`pattern_adversarial_noninterference`. The independent scorer returned exit 1.

The finalized verdict authenticates the unchanged original rejection at
`reports/m4_pattern_attempt04/verdict.json` (SHA-256
`ddbff364aa1b7bb9dce4db8b9273477a58dafff8f9d45e3740a275f49c26a7c5`).
The initial v7 scorer's `original_failure_preserved: false` was an input-error
marker, not evidence of loss; the finalizer records that correction and leaves
the decision and raw verdict unchanged.

### Outcomes and controls

| Arm / check | Development | Held-out | Regression / general | Meaning |
|---|---:|---:|---:|---|
| Direct pattern | 0/4 | 0/8 | 0/8 regression | No measured static utility |
| Router | 4/4 | 6/8 | 8/8 regression | All 20 delayed rows routed `fast`; not static attribution |
| Fast parent/candidate | 4/4 | 6/8 | 8/8 regression | Frozen fast behavior preserved |
| Linear control | 1/4 | 1/8 | 2/8 regression | Above direct pattern on dev/regression |
| Fast+pattern adversarial | 0/2 general-dev | — | 0/4 general | Below disabled 2/2 and 2/4; parent is 0/2 and 0/4 |

The historical no-warmup direct-pattern result was 0/4 development and 1/8
regression. The v7 direct arm is worse on regression and does not improve
development. General router rows explicitly selected `pattern` (4/4 general and
2/2 development routes), but their accuracy is not evidence of useful static
memory: general pattern accuracy is 0/4 and development pattern accuracy is 1/2.

No delayed router row had complete query-hash coverage (known/total ranged from
416/476 to 488/508). Thus the direct-pattern rows are valid all-row direct-arm
measurements, but **fully covered, actually routed pattern utility is
unmeasured**. The gate names must not be read as evidence that such cases were
present. Full-context general accuracy is 4/4, so the general items are
answerable.

`general_development_noninterference` passing means only “not below parent
adversarial”; it is not a disabled-base comparison. The pattern adversarial
check compares retention with disabled but retention-development with parent.
The raw fast+pattern results are below disabled on both splits regardless.

### Training, restart and independent checks

- Four source-LM updates consumed each of the 48 delayed-training observation
  chunks once; 16 delayed response-loss updates followed. The source phase used
  masked BOS prefixes, preserved the sense-target guard, saved a warmup
  extension, reset seed/optimizer, and saved the final extension: **20 updates,
  two checkpoints**. Source observation tokens are 1,412 (1,460 positions when
  the 48 BOS prefixes are included); delayed action/target tokens are 4,251/182.
- Update-two gradients are finite/nonzero; sparse AdamW behavior, disabled-base
  token parity, fast-module parity and all 20/20 correction checks pass.
- Same-process restart now reloads the saved extension and then the fast image
  with the current producer. State, lineage, extension manifest/weights, routed
  answer/token IDs and direct-pattern answer/token IDs all pass without source
  re-encoding. Fresh-process restart was **not run**, as declared for a rejected
  candidate.
- The independent scorer recomputed 427 generated answers and 1,579 new tokens;
  the six external parent adversarial rows are not counted as new generations.

The scorer re-scores preserved JSON; it does not regenerate answers. Actual
optimizer groups contain **34,615,300 pattern parameters**. The extension audit
reports 46,850,568 trainable parameters because pre-run trainability flags are
restored before saving; that larger number is not the v7 trained scope.

### Collision, latency and resource scope

The [independent audit](../../reports/m4_static_attempt07/collision_audit.json)
reused the production hash implementation and validated eight numeric fields
per group across all 51 training/router hash-call prompts and the separate
24-prompt training scope. It did not regenerate answers or provide a second
hash implementation. It found **zero distinct-n-gram collisions in both scopes**.
Repeated lookups are not collisions:

| Scope / order (per head) | Lookups | Distinct n-grams | Repeated lookups | True collision n-grams |
|---|---:|---:|---:|---:|
| All router calls / 2-gram | 3,181 | 220 | 2,961 | 0 |
| All router calls / 3-gram | 3,130 | 279 | 2,851 | 0 |
| Training only / 2-gram | 1,502 | 141 | 1,361 | 0 |
| Training only / 3-gram | 1,478 | 170 | 1,308 | 0 |

This is a hash-call audit, not a claim about every possible future pattern
forward. The report's `lookup_latency` field is a single-development-prompt,
three-call `pattern_port` forward microbenchmark: first call 1.978 ms and warm
minimum 1.846 ms (the better of only two subsequent calls). Embeddings, native
hidden-state computation, routing and generation are outside the timed region;
this is not end-to-end or process-cold latency.

Table parameters are 134,217,728 bytes; the dense-gradient upper bound
is the same; optimizer state is 269,527,088 bytes. Profiled wall time was
184.082 s (guard 186.466 s), sampled peak driver 11.158 GiB, minimum observed
free memory 74%, and swap growth 0. The run directory is 359 MB, including two
178.7 MB extension files; reports/verdict/audit are about 312 KB. No downloads
or native-base copies occurred.

### Preserved static-arm attempts and disposition

- `attempt01`: direct launch rejected because the capacity-preflight entrypoint
  was required; no model training.
- `attempt02`: source-LM labels violated the sense-prefix guard; fixed with a
  masked BOS prefix in v3.
- `attempt03`: training/evaluation completed but report assembly hit the missing
  collision variable; no completed verdict was inferred from it.
- `attempt04`: completed but used an invalid pattern-only adversarial control;
  it is not accepted evidence.
- `attempt05`: valid fast+pattern adversarial control and reject verdict, but
  collision/restart evidence was incomplete; retained as historical evidence,
  not used for the final collision/restart claims.
- `attempt06`: extension reload invalidated the already-loaded fast-image
  producer; v7 reloads the image after the extension.
- `attempt07`: complete v7 evidence and independent reject.

The original [M4 verdict](../../reports/m4_pattern_attempt04/verdict.json) and
sections 1–4 remain unchanged. M4 is not promoted, M5 is not admitted by fast
scores, and retention repair remains open. A future static arm needs genuinely
covered non-training cases and a new predeclared hypothesis; repeating this
exposed recipe to chase a pass is not justified.

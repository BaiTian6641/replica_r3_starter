# MiniCPM5 vision-collapse diagnostic

**26 October 2026 · attempt02 selected · engineering pass, learnability not
established.** The sealed training-only diagnostic compares a fresh seed-29
projector with the pinned train04 extension. The corrected protocol is at
`c595751`; the machine-readable receipt is
[summary.json](../../reports/vision_collapse_diagnostic_v1_20261026_attempt02/summary.json).
No development, held-out or semantic-audit case payload was read.

## What was measured

Both stages share the sealed MiniCPM source, 26 training rows (14 answers and
12 captions), frozen SigLIP features, prompt/label builders and controls. Only
six projector tensors differ; the other 43 extension tensors are unchanged.
All 26 fit rows pass prompt-mask, response-label, shift and EOS checks. There
are 208 control records (196 available plus 12 unavailable sibling placeholders)
and 144 comparison cells. All 196 available generations have first generated
token equal to the teacher-forced prefill argmax.

NLL below is the macro-average of per-row mean response NLL. Declared and
scored denominators are separate; sibling has six unavailable rows per stage.

| Stage | Control | Declared / scored | First-token gold | Response NLL | Unique generations |
|---|---|---:|---:|---:|---:|
| fresh | correct | 26 / 26 | 0/26 | 10.274771 | 26 |
| fresh | sibling | 26 / 20 | 0/20 | 9.959036 | 20 |
| fresh | shuffled | 26 / 26 | 0/26 | 10.379171 | 16 |
| fresh | no image | 26 / 26 | 1/26 | 9.746723 | 7 |
| train04 | correct | 26 / 26 | 5/26 | 1.517730 | 3 |
| train04 | sibling | 26 / 20 | 3/20 | 1.501631 | 3 |
| train04 | shuffled | 26 / 26 | 7/26 | 1.752725 | 4 |
| train04 | no image | 26 / 26 | 1/26 | 9.746723 | 7 |

Train04 therefore fits target text strongly relative to fresh, but this does
not establish correct visual dependence. Shuffled first-token accuracy exceeds
correct, sibling NLL is slightly lower than correct, and generations frequently
remain unchanged after image swaps. Train04 correct generations are only
`yes` (10), `blue` (10) and `a` (6). Correct and sibling outputs match for
16/20 rows and for all 10/10 available caption rows. Only 3/26 correct rows
generate the exact target; fresh has zero exact targets in every control.
Positive opposite-token gaps demonstrate some logit-level image sensitivity,
not robust image-specific generation.

## Feature evidence

The text-only ceiling is stage-invariant at 1/14 answer first tokens. A frozen-
encoder leave-one-pair-out readout reproduces 5/5 opposite-answer pairs, but
that is an internal readout over the same five training pairs, not external
held-out evidence or candidate-model quality.

| Metric | Fresh | Train04 |
|---|---:|---:|
| Projected RMS | 0.414244 | 1.410771 |
| Mean-patch / projected RMS | 0.675777 | 0.936880 |
| Mean-patch / text RMS | 17.6026 | 83.1180 |
| Sibling cosine distance | 0.023259 | 0.002565 |
| Cross-pair cosine distance | 0.047448 | 0.006315 |
| Within-image spatial variance | 0.093220 | 0.243201 |
| PCA content ratio | 0.374672 | 0.448277 |

Train04 is mean-dominated and reduces mean-feature cosine separation by about
9.1× for siblings and 7.5× across pairs. Increased absolute spatial variance
and PCA content mean this is **not** evidence of complete visual-information
collapse. The train04 PCA ratio is a 48-iteration, explicitly non-converged
approximation (`delta 7.4094e-08 > tolerance 9.9535e-10`); fresh converged at
iteration 29.

## Execution and independent review

The guarded MPS run completed in 268.01 s with one model load and two extension
loads. It used no downloads, CPU fallback, optimizer, backward pass or
checkpoint/extension writes. Sampled peak driver memory was 10,877,730,816
bytes, swap growth zero and minimum observed host free memory 75%. The final
artifact total is **6,005,936 bytes**: 6,003,939 across 11 listed files plus the
1,997-byte manifest. Telemetry is sampled; non-load counters are source-audited
rather than syscall-level attestation.

The affected CPU suite passes **123 tests**. Independent review confirmed
sealed-data access, deterministic controls, stage isolation, migrations, raw
metrics and all artifact hashes. It initially found four reporting defects:
hardcoded failure-load counters, ambiguous sibling denominators, incorrect PCA
convergence and manifest byte-scope wording. Attempt02 corrects all four;
closeout verdict is **correct / engineering pass** with 0.99 confidence.
Native01 and attempt02 rows, cells, fit evidence and protocol are byte-identical.
Full logits and projected patch tensors are not retained, so verification uses
the persisted raw metrics and aggregates rather than reconstructing forwards.

## Disposition and next gate

Fresh is operationally unaligned on these rows. Training-set text fitting
occurs, but robust image-dependent generation remains unestablished. No
held-out claim, V0 alignment or learned-quality promotion is authorized.

The single next justified experiment is a predeclared **read-only mean-only
projected-patch ablation** on the same sealed rows and pinned train04 stage.
Compare correct/sibling/shuffled opposite-token gaps and greedy swap-invariance
against attempt02 to test whether spatial residuals contribute image-specific
information beyond the dominant mean. No new training or evaluation-split
access is needed for that diagnostic.

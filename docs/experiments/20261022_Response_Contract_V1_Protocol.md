# One-line response contract v1: predeclared GPU baseline

Declared before native generation on 22 October 2026, America/Chicago.
The [machine-readable contract](../../configs/response_contract_v1.json) is the
instruction and settings authority. Its SHA256 is
`8f881ec8b365e70278e2478686a32595268875aec9e3cb470b0f8fda9c0c6735`.
This protocol, the prepared case/spec packet and executable code will be
committed before the one native run. No training or checkpoint save is planned.

## Preserved failure and hypothesis

The [local failure analysis](../../reports/response_contract_failure_analysis_20261022.json)
records the inspected completions, rendered template suffix and verified facts
before this intervention. [Offline source checks](../../reports/response_contract_source_reuse_20261022.json)
reverify all nine sealed source members and 1,037 prior experiment artifact pins.

The [previous independent result](../../reports/infrastructure_preflight_v2_20261022/native_B1_scoring.json)
has 0/16 task-correct and format-valid responses, all 16 at cap 32, no EOS and
15 distinct completions. Raw text spends the budget restating or explaining
the task. The original query specifies ask/UNKNOWN behavior but does not state
the complete canonical response grammar. The byte-verified source, FP32 MPS
placement, offline guard, prompt-only generation and template hashes passed.

The local chat template closes its thinking block before the answer when
`enable_thinking=False`; the observed prose follows that closed block. Disabling
thinking does not itself prohibit explanations in the answer. This diagnosis
comes from local rendered-template and token evidence, rather than an assumed
model behavior from another release.

There are also substantive warning signs. In `i2s41-de-fsm-001.q0`, the old
completion assumes START corresponds to S0 although START is absent. In the
missing-MASK register case, it produces a capability refusal. Thus output
instructions and the tight budget are plausible contributors, while a purely
formatting explanation is insufficient. No optimizer updates occurred in this
baseline, so this failure cannot diagnose the experimental Muon optimizer.
No current evidence isolates a hardware or source-corruption cause.

The test hypothesis is that a uniform explicit output/missing-value instruction
improves contract compliance at the unchanged cap. Correctness is evaluated
independently; compliance alone cannot establish successful reasoning.

## Intervention and fixed controls

Append the following common instruction to the canonical task prompt:

> Response contract v1: Output exactly one line, then end the response. No reasoning, explanation, Markdown, quotes, or prefix. Do not invent missing values.

Then state the permitted complete answer and conditional missing-field grammar,
using only the family and static contract. The full exact appendix is rendered
from the frozen JSON. Its four permitted answer forms are:

| Family | Complete-evidence output | Missing-name order |
| --- | --- | --- |
| Register XOR | Base-10 integer result | RULE, INPUT, MASK |
| Affine modulo 97 | Base-10 integer result | RULE, A, X, B |
| Finite state | S0, S1, S2 or S3 | RULE, START, SYMBOLS |
| Tool recovery | RETRY lookup, VERIFY receipt or STOP | RULE, STATUS, ATTEMPT, LIMIT |

For missing items, the original task still chooses `UNKNOWN` or
`Please provide <fields>.`; the names follow the static order and use comma-space
separation. Do not select instructions using a gold action, response, observed
value or missingness. The new flattened view is an explicitly declared prompt
projection: canonical prompt, two newlines, family appendix. Canonical episodes,
raw source bytes, evidence text/order, IDs, labels and old exports stay unchanged.
No oracle annotation, worked answer example or forced decoder grammar is added.

Keep the original **16 development worlds/cases in the same order**, same sealed
MiniCPM5-2B revision `f97400052a43d642bbc6e9975e2397e3ae6a6b52`, local tokenizer
and chat template, MPS FP32, CPU fallback disabled, thinking disabled, greedy
decoding, one beam, seed 17 and **cap 32**. Reuse
`checkpoints/minicpm5_source` offline; do not acquire another original. Native
prompt/reference length must remain within 512 with zero truncation. Record
added prompt tokens, since instructions increase inference work.

The prepared [evaluation specification](../../reports/response_contract_v1_20261022/evaluation_spec.json)
and [token audit](../../reports/response_contract_v1_20261022/token_accounting.json)
freeze the same ordered cases. Total prompt tokens increase from 3,946 to
5,574 (1,628 additional tokens); maximum prompt-plus-reference length increases
from 324 to 429. No truncation occurs. Supervised answer/EOS and source-writer
views are unchanged. This is an instruction intervention with increased prompt
work, not an equal-total-compute comparison.

The original strict parser and independent raw-source solver remain unchanged.
The new scorer reconstructs the declared prompt projection and rechecks token
IDs, full completions, prompt/template hashes, EOS/caps, source truth and the
previous raw baseline. It does not extract an answer from explanatory prose.
The old parser trims surrounding whitespace and accepts signed equivalent
integers; preserve those semantics and report exact text separately.

## Outcomes, stopping and interpretation

Primary contract outcomes are unchanged grammar validity, separate single-line
compliance (nonempty after surrounding whitespace trim, no internal CR/LF),
their conjunction, EOS and caps. The format gate requires 16/16 compliant
grammatical responses, all EOS and no caps. Report strict task correctness,
action/decision/value errors and per-family results independently. Keep every
raw answer, including format-valid wrong answers and residual refusals.

The paired description compares the old and new exposed development cases.
Any change estimates this instruction bundle on these reused cases; it does not
identify which sentence helped, prove broad generalization or justify model
promotion. All mechanisms remain shared across roles, and host code supplies
the active evidence snapshot. This is not autonomous temporal retrieval.

Run exactly **one** fresh frozen-original job over all 16 cases. Do not adjust
the cap, parser or prompt after seeing its answers. If the format gate fails,
preserve the failure and predeclare a separate future diagnosis. If formatting
improves but task errors persist, classify their evidence/semantic causes before
training comparisons. No adaptive follow-up arm is included in this protocol.

Use the existing sampled memory guard: driver limit 40 GiB, swap-growth limit
512 MiB, host-free minimum 15% sustained for three samples, driver cadence .25 s
and host cadence 2 s. Replay its traces independently. Report MPS live/driver
and RSS separately because counters overlap and samples can miss transients.
Archive small raw results, with zero learned-weight payloads.

S1.3 public admission, the remaining S1.4 evidence/retention/calibration controls
and S1.5 still precede S2.4. This bounded baseline does not start an architecture
training campaign. Preserve the existing untracked runtime draft and all selected
D0/D1/D2 checkpoints.

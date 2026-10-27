# NL1 benchmark validity correction

**26 October 2026 · v1 findings reproduced; v2 grammar, fixture and protocol
implemented. Native v2 initialization/evaluation is not yet a result.** The
hashed-routing experiment and all v1 scores remain unchanged as negative
controls. This document separates measurement correction from learned utility.

## 1. Reproduced v1 defects

The read-only audit is
[reproduction.json](../../reports/nested_text_validity_20261026/reproduction.json),
repeated byte-for-byte in `reproduction_repeat.json`. It uses the sealed
tokenizer only, reads no sealed-final rows and executes no model.

| Finding | Existing evidence |
|---|---|
| Prompt-hash routing loses the taught concept | 0/16 development queries route to an expert taught that concept; paraphrase changes the route |
| Exact-string scoring rejects correct natural answers | `8 times 7 is **56**.`, `The capital of France is Paris.`, `The square of 9 is 81`, and `A leap year has **366 days**.` all score false |
| Eight-token generation is inadequate | 194/196 outputs hit the cap; several cap-hit answers are visibly incomplete. Cap hits alone are not all classified as truncation |
| Sentence-completion fixture is invalid | `legacy-0002` supplies `closes` inside the prompt and has no blank |
| Processed positions are misreported | Two feature passes process 1,337 positions, but cost.json reports 117 response-supervised tokens as processed positions |
| Duty-cycle denominator is wrong for the gate | Teaching-only `update_seconds / teaching_wall_seconds = 0.999938` is compared with an amortized workload gate that has no workload definition |
| Evaluation latency omits work | The timer starts after plastic feature sensing and `MemorySession.prepare`, so it measures generation only |

The position breakdown is 610 prompt-sensing positions plus 727 full
prompt+response positions, totaling 1,337 across 54 feature forward calls.
There are 117 response-supervised tokens, including EOS. These are different
units.

## 2. Versioned correction

### Fixture and routing

`configs/nested_text_episodes_v2.json` changes exactly one row:
`legacy-0002` becomes `Complete the sentence: The library ____ at dusk.` with
expected answer `closes`. Its prompt digest is updated; every other episode row
is byte-identical. The v1 generator output and manifest remain byte-identical.
The new manifest is `8bf2e64f…` under `data/nested_text_nl1_v2_20261026/`.

`configs/nested_text_nl1_v2.json` declares **one shared expert**. Fixed routing
is constant by construction and cannot select different experts for a
paraphrase. No evaluation labels, world/family labels or gold record IDs enter
routing. The v1 16-expert prompt hash remains a negative control.

### Response grammar and stopping

`src/replica_lab/nested_text_scoring.py` implements
`replica.nested_text.scoring.v2`:

- Bare expected answers are accepted with optional terminal punctuation and
  Markdown emphasis.
- Complete declaratives containing the expected whole phrase are accepted.
- Questions, uncertainty, refusals, negation/contradiction, competing candidate
  answers and visibly incomplete text are rejected.
- `unknown` and `revoked` are exact status responses only.
- Native runs retain raw generated IDs, content IDs, stop token/reason and
  actual token-budget truncation. Empty or truncated output receives no credit.
- Generation stops at the first complete sentence boundary, EOS `1` or
  `<|im_end|>` `130073`, or the 24-token hard cap. Sentence completion is not
  truncation; a token-budget ending receives no credit. The host predicate
  never receives the expected answer and cannot manufacture a stop receipt.

The scorer is a narrow grammar, not a general semantic-equivalence claim.
Positive, incorrect, ambiguous, contradictory, truncated, status and stopping
contracts pass **42 focused tests**.

### Corrected retention replay

[scoring_v2_replay.json](../../reports/nested_text_validity_20261026/scoring_v2_replay.json)
applies v2 to the preserved 196 dev04 outputs without a model run or inferred
stop metadata. It does not rewrite dev03/dev04 or claim new learning.

| Measurement | Old exact-string | V2 replay |
|---|---:|---:|
| Preteach legacy retention | 0/8 | 4/8 |
| Taught legacy retention | 0/8 | 4/8 |
| Retention decline | 0.00 | 0.00 |

New-instance, correction and unaffected answers remain 0. This establishes a
meaningful preteaching baseline for the corrected grammar; it is not evidence
of learned improvement.

### Workload and latency

The declared development workload contains 196 attempts: 60 taught primary, 60
preteach primary, 16 shuffled, 16 reset-P, 16 spine-absent, 12 wrong-world-P and
16 wrong-world-on-primary. Sealed-final access is not part of the workload.

The duty ratio is:

$$\frac{\text{learning update seconds}}
{\text{teaching end-to-end seconds + evaluation end-to-end seconds}}$$

End-to-end time includes model/protocol initialization, sensing, request
preparation, generation/finalization, scoring, checkpoint/export and reporting.
No idle time is inserted to satisfy the ratio. A teaching-only ratio cannot
satisfy the amortized gate.

Per-request timing separately reports plastic-read preparation, session
preparation, generation/finalization, release and total. Position accounting
separately counts plastic feature prompts, session-sense prompts, generation
prefill, decode positions, actual forward/backward calls and response-supervised
tokens. Replay count is zero for this baseline.

## 3. Commands and current disposition

```bash
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 .venv/bin/python \
  scripts/audit_nested_text_validity.py \
  --output reports/nested_text_validity_20261026/reproduction.json

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  scripts/replay_nested_text_scoring_v2.py \
  --output reports/nested_text_validity_20261026/scoring_v2_replay.json

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_nested_text_scoring.py tests/test_nested_text.py
```

**Designed:** one-expert behavioral baseline, amortized workload accounting and
full phase instrumentation. **Implemented:** scorer, stopping metadata,
minimal fixture correction, protocols and replay tools. **Executed:** tokenizer
audit and corrected replay only. **Not executed or qualified:** native v2
initialization/evaluation, behavioral response-CE training, independent
uninterrupted/resumed continuation, held-out evaluation and any capability
promotion.

The next gate is zero-update initialization followed by development-only P-only
evaluation with the v2 stop, latency and accounting evidence. A separate
frozen-decoder behavioral-reference protocol is required before any teaching.

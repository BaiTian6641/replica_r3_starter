# Response readout diagnostic and rejected budget02 probe

**26 October 2026 · read-only diagnostic completed; optimization signal is
partial, exact response fit remains zero; the proposed two-step arm is rejected
before teaching.** The diagnostic and original budget02 declaration are committed
at `65c20d7`; one-clock validation and the explicit rejection are `56ce435`.
This follow-up does not establish a useful behavioral baseline or authorize
held-out evaluation.

## Diagnostic declaration and lineage

`configs/nested_text_response_diagnostic_v1.json` fixes a 28-row read-only
comparison between the completed response-CE state and its null parent. The
sense frame is prompt-only. Gold response tokens enter only a no-grad
teacher-forced action-pass loss; they never enter request construction, an
optimizer, or a generation control. The existing greedy fit report is reused;
no new generations are produced.

| Binding | Value |
|---|---|
| Diagnostic configuration SHA-256 | `e46bb3f14c40c2edae38767ee5139886220e47edf886d1367de1452170e05ac9` |
| Parent behavioral protocol SHA-256 | `053ed39a08bfa55e09b36314cc1bb4ae9c55308d774581bd6d3886ab94e4c14e` |
| Trained P-only state | `runs/nested_text_behavioral_reference_attempt03_20261026/teach_primary/p_only/state.pt` |
| Trained state SHA-256 | `abf2bf5df2c4b5f4a6e2b9cd3e4031b3febafd1d78dbc5993d9954d14b739efb` |
| Null-parent P-only SHA-256 | `84b98f60f5f97f27fbf045c6e1f97d5424b7b8ffa3aee34cc2189d6665b25e45` |
| Reused greedy report SHA-256 | `48c0e811f849d08b9251a76e45fdd0a655d50eed70c2c64362672855be11b5e2` |
| Raw diagnostic report | `runs/nested_text_response_diagnostic_attempt01_20261026/diagnostic/response_diagnostic.json` |

The command was:

```bash
.venv/bin/python scripts/run_capacity_preflight.py \
  --entry-point diagnose_nested_response.py \
  --report runs/nested_text_response_diagnostic_attempt01_20261026/guard.json -- \
  --protocol configs/nested_text_response_diagnostic_v1.json \
  --episodes data/nested_text_nl1_v2_20261026 \
  --model-path checkpoints/minicpm5_source \
  --model-lock configs/minicpm5_sealed_lock.local.json \
  --output runs/nested_text_response_diagnostic_attempt01_20261026/diagnostic \
  --device mps
```

Both states used the same 28 prompts and response rows, one loaded sealed
backbone, strict P-only weights/bindings, and separate empty ledgers. All 56
requests were source-free, with no controller, coordinator or updates. The
manifest's sealed-final file is hashed for lineage by the shared loader; no
sealed-final rows were parsed or scored and `heldout_opened=false`.

## Measured readout result

| Metric | Null parent | Trained response-CE state | Trained minus baseline |
|---|---:|---:|---:|
| Mean teacher-forced response NLL | 10.689084 | 5.608277 | -5.080808 |
| Response-token accuracy | 1.6393% | 13.9344% | +12.2951 percentage points |
| Exact teacher-forced response | 0/28 | 0/28 | 0 |
| Successful requests | 28/28 | 28/28 | — |
| Maximum model-loss/aligned-NLL difference | 9.54e-7 | 4.77e-7 | — |

The reused greedy fit result is **0/28** with 335 generated tokens, two
budget-truncated attempts and no scoring/request failures. Teacher-forced exact
response and greedy scored answer are different readout conditions; no
cross-condition accuracy delta is claimed.

The trained state therefore changes the response distribution in the expected
optimization direction, but the signal is incomplete: neither exact
teacher-forced responses nor greedy answers fit the teaching rows. This is not
a decoding-only diagnosis, calibrated confidence, useful memory, or a capability
claim. Raw per-row target IDs, predicted IDs, decoded text, losses and request
audits are preserved in `response_diagnostic.json`.

### Diagnostic cost

The diagnostic took 13.386874 s end-to-end. Each state performed 84 attributed
forward calls, 2,033 processed positions and 122 supervised response-token
positions; backward calls, updates and new generated IDs were all zero. One
model and tokenizer load served both states. The watchdog peak was
10,824,384,512 bytes, with zero observed swap growth and 76% minimum observed
host free memory. The earlier 29.049763 s greedy fit evaluation is a separate
cost and is not included in this diagnostic's runtime.

## Teaching-accounting correction

The original response-CE `cost.json` remains byte-preserved. Its
`processed_positions=2670` omitted the detached response-target extraction's
full-sequence positions. The corrected derived total is **3,429** for the same
28 one-step events:

```text
637 detached prompt + 759 detached target + 637 differentiable-read prompt
+ 637 session-sense prompt + 759 response-CE positions = 3,429
```

The derivation, source-cost hash and unchanged forward/backward counts are in
`runs/nested_text_behavioral_reference_attempt03_20261026/teach_primary/corrected_accounting.json`.
`response_teaching_work()` now counts detached targets once and read/sense/CE
work per optimizer step, including multi-step inputs for validation purposes.
No model output, checkpoint or fit score changed.

## Budget02 rejection

The follow-up `configs/nested_text_behavioral_budget02_v1.json` was committed
before native initialization and proposed two optimizer steps per event (56
primary steps) while leaving data, objective, optimizer, read path and gates
unchanged. It was intended as the only additional optimization-budget arm.

Initialization completed with zero events and zero updates at
`runs/nested_text_behavioral_budget02_20261026/init_primary/`. Inspection then
found that the declaration requested `steps_per_event=2` while the saved parent
contained `steps_per_event=1`. `TextProtocol` explicitly requires exactly one
step per event; the one-clock contract was not weakened to accommodate the
probe. Initialization used configuration SHA-256
`e581a4fc025794e6f5f6889df18ed66ab2ea4770ffd8bf773cded64be96237f6`.

No teaching, fit evaluation, wrong-world training, development or held-out
scoring followed. The mismatch and zero-update disposition are preserved in
`init_primary/rejection.json`. The current configuration is marked
`rejected_steps_per_event_invariant`, with native execution and training
closed. The validator added in `56ce435` now rejects multi-step declarations
before model loading. The original declaration remains available in commit
`65c20d7`; the current rejection does not rewrite that history.

The initialization command, which did not teach, was:

```bash
.venv/bin/python scripts/run_capacity_preflight.py \
  --entry-point run_nested_text.py \
  --report runs/nested_text_behavioral_budget02_20261026/init_primary.guard.json -- \
  --mode init --protocol configs/nested_text_behavioral_budget02_v1.json \
  --episodes data/nested_text_nl1_v2_20261026 \
  --model-path checkpoints/minicpm5_source \
  --model-lock configs/minicpm5_sealed_lock.local.json \
  --output runs/nested_text_behavioral_budget02_20261026/init_primary \
  --device mps --world-role primary
```

## Verification and disposition

The current focused suite passes **200 tests** across nested text, nested
learning, scoring, architecture adapter, memory session and matched memory
coverage. It includes a real tiny-native-session teacher-forced test with exact
loss alignment, prompt-only/source-free requests and zero gradients, plus
multi-step accounting and one-clock rejection tests. The earlier full
repository run passed 2,362 tests plus 21 subtests with 8 skips and 7
Spark/vision pin/receipt failures outside this increment; it was not rerun
after this follow-up's focused changes.

**Implemented:** reusable read-only teacher-forced diagnostic, matched
null-parent control, existing-greedy comparison, exact response-position
accounting and fail-closed one-clock validation. **Measured:** trained/null
NLL, token accuracy, exact-response counts, source audits, runtime and memory.
**Rejected:** exact teaching-fit utility and the invalid two-step budget probe.
**Unestablished:** a useful simple plastic-learning baseline. **Unrun:**
budget02 teaching/fit, development, wrong-world follow-up, held-out evaluation,
multi-seed promotion and any new objective or backbone update.

A further arm would need a separately declared one-clock-compatible hypothesis
or an explicitly authorized protocol/architecture change. Repeating the same
training recipe or opening sealed-final rows is not justified by this result.

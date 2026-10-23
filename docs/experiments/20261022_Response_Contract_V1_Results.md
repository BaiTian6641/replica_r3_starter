# Response contract v1: GPU results and residual error review

22 October 2026, America/Chicago. The predeclared one-line instruction changes
termination behavior, but the frozen original still scores **0/16 task-correct**.
All 16 responses now end with EOS below the unchanged 32-token cap. Only one
response satisfies the canonical grammar, and that response names the wrong
missing fields. The declared format gate fails; no training comparison or model
promotion follows from this run.

The [protocol](20261022_Response_Contract_V1_Protocol.md),
[contract](../../configs/response_contract_v1.json), code and prepared inputs were
committed as `430e2986fbc8ce8010990067bb9bd1b8bc89e142` before generation.
The [launch receipt](../../reports/response_contract_v1_20261022/native_launch.json)
checks committed input bytes. Exactly one fresh native job used the same 16
ordered development cases, source evidence, reference answers, sealed original
MiniCPM5-2B, tokenizer/template, greedy decoding, thinking disabled, seed 17,
MPS FP32 and cap 32. The deterministic family appendix was the intervention;
neither observed missingness nor expected labels selected its wording.

## Measured outputs

The [independent scoring report](../../reports/response_contract_v1_20261022/native_B1_scoring.json)
reconstructs raw-source truth, prompt and output token identities, the declared
projection and the preserved baseline. The strict whole-response parser remains
unchanged. It does not extract `UNKNOWN` from malformed text. Single-line
compliance trims surrounding whitespace and is separate from grammar validity.

| Outcome | Original prompt | Contract v1 |
| --- | ---: | ---: |
| Task-correct | 0/16 | 0/16 |
| Canonical grammar valid | 0/16 | 1/16 |
| EOS | 0/16 | 16/16 |
| Cap hits | 16/16 | 0/16 |
| Generated tokens, including EOS when present | 512 | 194 |
| Generation time | 13.151 s | 6.681 s |

All new responses satisfy the separate single-line metric and terminate after
9–17 generated tokens including EOS. Each family remains 0/4 task-correct.
The error breakdown is 15 malformed responses and one grammatical wrong value.
The full response list stays in the scoring report and raw archive; failures
remain failures under the predeclared rule.

Register XOR produces `UNKNOWN, INPUT, MASK` on all four cases. Tool recovery
produces `UNKNOWN, Please provide RULE, STATUS, ATTEMPT, LIMIT` on all four.
These repetitions cross complete-evidence, ask and unknown cases. They show
family-level repetition rather than a return to the earlier global constant-zero
output. Fifteen responses contain `UNKNOWN`; seven also contain `Please provide`.

The sole grammar-valid response, for `i2s41-de-fsm-001.q0`, is
`Please provide RULE, START, SYMBOLS.`. Its observed snapshot contains the rule
and `SYMBOLS="bab"`; only START is unavailable at cutoff 5. The correct response
is `Please provide START.`. Correct ask action and decision type therefore do
not supply value or task correctness.

## What the evidence supports

The [residual review](../../reports/response_contract_residual_error_review_20261022.json)
independently replays all 16 raw source files and event byte spans without
importing the generator or infrastructure solver. It reconstructs observation
cutoffs, corrections and active facts, then calculates XOR, affine residues,
state paths or recovery-policy decisions. All reconstructed answers and support
events match the frozen evaluation. Eight cases have complete evidence; the
other eight each lack exactly one fact. Every case has an observed rule. The
review records the facts, missing items, operation traces and raw completions.

The paired observation supports a change from capped explanatory prose to short
terminated outputs after adding the instruction bundle. Correctness stays zero.
A budget-only explanation is inadequate for these residual errors: every new
response voluntarily terminates below 32 tokens. The original failure also
included an unsupported START=S0 assumption, so semantic and evidence-use
warnings preceded this intervention.

Copying a fallback form, confusing the `UNKNOWN`/ask branches, and failing to bind
observed JSON facts to required fields are plausible hypotheses. The outputs
alone do not establish an internal mechanism or isolate which instruction
sentence matters. The known one-line result demonstrates missing-field selection
error; malformed responses cannot be assigned an unstated intended answer.
This frozen baseline has no optimizer updates, so it does not diagnose Muon.

The local upstream [README](../../checkpoints/minicpm5_source/README.md#L283)
recommends sampling at temperature 1.0 and top-p 0.95, and suggests repetition
penalty 1.05 when repetition occurs. Its quickstart enables thinking. The
[pinned official copy](https://huggingface.co/openbmb/MiniCPM5-2B/blob/f97400052a43d642bbc6e9975e2397e3ae6a6b52/README.md)
and local README hash are recorded in the residual review. These differences
suggest a separate future inference-settings qualification; they do not prove
the cause here. Greedy decoding and thinking disabled remained fixed to preserve
the v1 comparison. The local configuration specifies ordinary `LlamaForCausalLM`
without an `auto_map`; no alternate custom architecture was substituted.

## GPU cost and preservation

The [token audit](../../reports/response_contract_v1_20261022/token_accounting.json)
records 5,574 prompt tokens versus 3,946 previously: 1,628 added tokens. Maximum
prompt-plus-reference length rises from 324 to 429, within 512 without truncation.
Reference/EOS supervision and the separate source-writer view are unchanged.
More prompt tokens and fewer output tokens make this an instruction comparison,
not an equal-total-compute speed benchmark.

| Measured v1 execution | Result |
| --- | ---: |
| Synchronized generation / profiler / guard lifetime | 6.681 / 11.359 / 13.741 s |
| Sampled peak MPS live / driver | 9.426 / 10.104 GiB |
| Sampled RSS / process-lifetime RSS peak | 3.467 / 14.490 GiB |
| Observed guard swap growth | 0 bytes |
| Minimum sampled host free percentage | 71% |

GiB means bytes divided by 2³⁰. RSS and MPS counters overlap and must not be added.
MPS samples use .25 s intervals; host/RSS use 2 s. Samples can miss transients,
and process-lifetime RSS is a high-water mark. The guard replays exactly and
passes its 40 GiB driver, 512 MiB swap-growth and sustained 15% host-free limits.
Model parameters execute on the M2 Ultra GPU with CPU fallback disabled; Neural
Engine execution is unqualified.

The [experiment receipt](../../reports/response_contract_experiment_20261022.json)
binds the final archive, reuse evidence and validation. The original failed
packet, v1 protocol and selected checkpoints are preserved. The existing local
model is reused offline, with no downloads, optimizer updates or new checkpoints.
The [code validation](../../reports/validation_response_contract_code_20261022.json)
records 671 passing tests and one restricted-context MPS skip; the native job
above separately executes on MPS.

## Next development step

The v1 stopping rule is complete. No adaptive follow-up or training run belongs
to this protocol. A future v2 proposal should remove the combined OR instruction:
choose one missing-response policy using only the public original query wording
and family, while keeping available evidence distinct from response placeholders.
Do not choose it from a gold action, answer or observed missingness. Freeze that
proposal and its controlled settings before another native run. A later sampling
or thinking comparison would need its own declared contrast; changing all settings
together would obscure the reason for any improvement.

S1.4 remains incomplete, alongside S1.3 admission and S1.5 preparation. The
remaining evidence sensitivity, retention and calibration checks still precede
S2.4 architecture training. Sixteen reused exposed development worlds cannot
establish generalization, architecture gain or model promotion.

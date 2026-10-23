# Response policy v2: GPU baseline and error analysis

22 October 2026, America/Chicago. The [protocol](20261022_Response_Policy_V2_Protocol.md),
exact inputs, code and semantic-judge rubric were frozen in `b2010c2` before
one original-model M2 Ultra GPU run. **Strict correctness is 4/16; every success
is an UNKNOWN case. An always-UNKNOWN control would also score 4/16.**
This is a preserved development negative, not useful reasoning or architecture
gain. No second v2 native run followed the result.

The [independent source-bound scoring](../../reports/response_policy_v2_20261022/native_B1_scoring.json)
uses the unchanged strict parser. A fresh-context subagent independently read
the observations and derived each answer without gold labels, programmatic
solver grades or prior outputs. Its [immutable semantic review](../../reports/response_policy_v2_20261022/judge_semantic_raw.json)
also finds **4/16 correct, 12/16 incorrect, zero ambiguous**. Case-level
reconciliation is recorded separately in the
[judge comparison](../../reports/response_policy_v2_20261022/judge_reconciliation.json).

## What changed and what the scores mean

Only the appendix changed: each query receives one missing-response policy,
chosen from its public wording and family. Required items are listed separately
from answer placeholders. The sixteen fixed worlds, source evidence/order,
target labels, original model, tokenizer/template, greedy seed 17, thinking
disabled, FP32 MPS, CPU fallback disabled, cap 32 and ceiling 512 stay fixed.

| Measured outcome | Original | Contract v1 | Policy v2 |
| --- | ---: | ---: | ---: |
| Strict task correct | 0/16 | 0/16 | 4/16 |
| Whole-response grammar valid | 0/16 | 1/16 | 12/16 |
| Ends with EOS | 0/16 | 16/16 | 16/16 |
| Hits token cap | 16/16 | 0/16 | 0/16 |

All v2 outputs are single-line. Single-line form, legal response grammar and
evidence-based correctness are separate checks. UNKNOWN is grammatically valid
on the twelve queries whose missing policy permits it, even when their complete
evidence requires an answer. The semantic judge counts that conditional
instruction failure separately: only four responses are instruction compliant
and grounded. There is no semantic correctness disagreement to resolve by
relaxing the parser. The registered all-sixteen format gate still fails.

The [original capped-prose result](20261022_Infrastructure_Slice_V2.md) and
[v1 negative](20261022_Response_Contract_V1_Results.md), including raw answers,
hashes, guards and profiles, remain preserved. Reusing exposed development
worlds across three prompt interventions is development search; it is not
independent confirmation or an equal-total-compute comparison.

## Error pattern and interpretation

- **Complete evidence: 0/8 correct.** Every completion is UNKNOWN despite all
  required rule/facts being observed by the cutoff. No numeric/state/tool result
  is produced. The failures occur across all four families.
- **Missing evidence with UNKNOWN policy: 4/4 correct.** These are the only
  successes and explain the entire aggregate score.
- **Missing evidence with ask policy: 0/4 correct.** Requests include already
  observed fields. XOR asks for RULE and INPUT while omitting missing MASK;
  affine asks for A and X alongside missing B; tool recovery adds ATTEMPT and
  LIMIT alongside missing STATUS. FSM retains NAME and lists observed RULE and
  SYMBOLS alongside missing START. These are material evidence errors in
  addition to punctuation/placeholder failures.

The twelve UNKNOWN outputs and four request outputs follow the supplied
missing-policy split, while failing to establish whether evidence is complete.
This supports a narrower diagnosis: the simpler contract reduces response-form
mixing, but the model still fails conditional evidence use on this probe.
Copying/salience of the fallback branch is a hypothesis, not an identified
internal mechanism. The result does not isolate an arithmetic, chronology,
parser-reading or instruction-processing cause. Additional output budget alone
cannot explain short, completed wrong answers. No optimizer update occurred,
so the experiment does not diagnose Muon 8-bit.

## Resources and validation

The sealed local original MiniCPM5-2B revision
`f97400052a43d642bbc6e9975e2397e3ae6a6b52` is reused offline. There are
**zero downloads, optimizer updates or new checkpoints**. The selected
D0/D1/D2 artifacts and separate runtime draft are preserved.

V2 uses 5,294 prompt tokens and generates 86 tokens. Maximum
prompt-plus-reference length is 407; no row is truncated. Generation takes
**4.412 s** and the profiled lifetime is **9.031 s**. Sampled peak MPS
live/driver memory is **9.427 / 10.096 GiB**. Exact guard replay passes with
zero observed swap growth and minimum host-free 71%. These counters overlap
and must not be added; sampling can miss transient peaks. Native inference
executes on MPS GPU; Neural Engine execution is not established.

Initial scoring stopped because archived historical guards retain their original
run-directory binding. The [preserved validation issue](../../reports/response_policy_v2_20261022/initial_scoring_validation_issue.json)
and supplementary archive-reader repair are recorded in the scoring report.
The repair verifies all **14 historical files byte-for-byte** before resolving
their recorded original paths and replaying guards. Its **21 focused tests**
cover that boundary. The frozen specification/parser/scorer, current v2 guard
validation and model grades are unchanged; no native rerun is required.

The [final software suite](../../reports/validation_response_policy_v2_final_20261022.json)
passes **819 tests, one restricted-context MPS skip** (108.57 s wall time).
The pre-native suite remains recorded separately as 798 passed/one skip.
Software validation does not pass the failed native task/format gate.

## Architecture progress and next boundary

The separate [early S3.1 host contract](20261022_Host_Action_Contract.md) passes
**38 focused tests and 12 actual host-fixture checks** in its
[report](../../reports/host_action_contract_20261022/report.json). Host-owned
observations feed an actual pure-XOR tool; only a matching successful receipt
from the current epoch authorizes an answer. Revocation precedes cached-result
reuse. Restart checks a journal against a caller-retained trusted head without
reexecuting the tool. This is integrity relative to that head, not source
authentication, historical erasure or neural cache invalidation.

That package restores a static goal/action trace, not evolving goals/subgoals
or learned task continuation. It has no model proposer or experience-to-learning
path. Next separately predeclare source-grounding/action-prompt qualification
and one budgeted frozen-original MPS proposal → host/tool receipt → answer
pilot. Keep every failed action. S1.4 remains partial, S1.5 admission and S2.4
training remain blocked, and full selected-system S3.1 integration still
requires S2.5. No model is promoted by either the agent judgment or host fixtures.

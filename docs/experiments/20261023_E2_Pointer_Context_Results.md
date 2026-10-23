# E2 pointer context scorer: 12/12 on fresh values (opt-in diagnostic)

23 October 2026 UTC. No pointer training. A 6-parameter candidate-context head
was trained on top-record whole-word spans from existing training cases; the
published pointer weights stayed frozen. Evaluation used a new 12-case family.

## 1. Mechanism

For each top-record whole-word candidate, the context head receives only
source/query positional features:

- relative start and end;
- whether the span immediately follows `is`/`are`/`was`/`were`;
- span token length;
- normalized distance from the preceding linking verb.

It adds a scalar bias to the frozen pointer score. The host enumerates all
whole-word spans; the model selects the span. No expected answer enters the
features.

## 2. Result

| Arm | Fresh exact |
|---|---:|
| Frozen pointer, top record | **9/12** |
| Pointer + context head | **12/12** |

The context head fixed the previous entity-name misses (`Glacier`, `Jasper`,
`backup`) and selected the full value in all 12 cases. Training loss over 300
steps was ~**0.151 → 0.148**; the head has only weight[1,5] + bias.

The artifact is a diagnostic
`runs/e2_pointer_context_attempt04/context_head.safetensors`
(sha256 `d9dc540a…`). It is **not yet loaded by production
`MemoryAnswerService`**, not published as an active opt-in manifest and not a
default. Exact text remains the default.

## 3. Limits

The evaluation family is synthetic and freshly separated from the context
head's training cases, but the feature set is task-template-specific. The result
does not establish general pointer/context capability. A production arm must
integrate the head into the answer service, save a versioned artifact, repeat on
another fresh family and verify rollback before promotion.

## 4. Costs

**15.5 s**, **11.0 GiB** sampled peak driver, one model load, no pointer
optimizer update, no base copy or checkpoint.
Evidence: [report](../../reports/e2_pointer_context_attempt04/run_report.json).

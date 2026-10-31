# Spark paired-joint versus sequential-update diagnostic v18

Status: **selected and predeclared; not implemented, authorized or launched.** This is the only controlled training comparison selected from the v17 interference evidence. It is a fitting-only falsification diagnostic, not a candidate promotion or a revival of the falsified v16 content-key mechanism.

## Evidence and hypothesis

V17 attempt05 restored the same v16 terminal state before eight isolated single-world updates. Target expected-minus-alternative total-sequence margin improved in 7/8 arms, while the untouched partner's margin worsened in 7/8. Gradient cosine was positive in two of four pairs, so gradient opposition alone does not explain the outcome.

The single hypothesis is: **including both counterfactual worlds in each optimization step reduces pair-level margin interference relative to sequential single-world updates at equal per-world supervised exposure.**

No conclusion about generalization, useful recall, persistent learning or a universal interference law is permitted.

## Fixed parent and data

- Official sealed BF16 Spark source, revision `66e3ae7bee54e0a523d692cb8ef841f35409807b`; original parameters remain frozen and hash-checked.
- Parent extension and optimizer: `reports/intrinsic_memory_content_key_v16/native_bf16_two_attempt01/checkpoint`, manifest SHA-256 `a547053a30c58ad2c431b7a3e8a153f543f718a4328cf54f84c74d7acfec085c`, weights SHA-256 `66f9cec6ea2068a2432b055578039c716071a6bd6b0a3ba3ec97203e5cf40956`, optimizer SHA-256 `2ed73fa65963ada7182e5749937c6f73767f224a51de4cdb4ce9d0835ab2178b`.
- Fitting pairs only: IDs **0, 1, 4, 5**, worlds `v11_fit_00/01`, `02/03`, `08/09`, `10/11`. Fit file SHA-256 `106784510b4c26371c8a1f81736eea92443207d6028f63f1b491df0430dff92c`.
- Content-only keys, ordered values, source features, four slots, sites 3/7, post-block placement, NativeFusedAdamW, learning rate 0.0001 and initialization/execution seed 17 remain unchanged.
- Development rows and labels, generated answers, mechanism changes, compression fallbacks, exact-evidence retrieval and all later learning stages are excluded.

## One comparison

Both arms start from independently restored, identical parameter, optimizer-moment and CPU/MPS RNG states. Each arm processes pairs in order 0, 1, 4, 5 and presents each world exactly once.

| Arm | Per-pair optimization | Optimizer steps per pair | Supervised examples per pair |
|---|---|---:|---:|
| Sequential reference | One complete-response update on each world; pairs 0/4 use left→right and pairs 1/5 use right→left | 2 | 2 |
| Paired joint | One update using the mean of the two worlds' complete-response negative log-likelihoods | 1 | 2 |

The joint loss is the unweighted mean of the two per-world mean negative total log-likelihoods. Each per-world loss includes its answer tokens and one EOS and is divided by its actual supervised-token count. No host-selected answer, future observation or alternative response enters the source/sense path.

The arms match initial state, data, per-world exposure, supervised examples and per-world forward/backward work. **Optimizer steps intentionally differ (8 sequential versus 4 joint across the fixture).** Batching and update count are therefore the combined treatment; the experiment cannot attribute an effect to batch size alone.

## Measurements

For each world and arm, preserve expected and alternative answer/EOS likelihoods using each candidate's own tokenization under the same memory context. Define total margin as expected total log-likelihood minus alternative total log-likelihood.

Record:

- Baseline, intermediate and final margins for every world; per-pair minimum margin and changes from the common baseline.
- Every sequential target/partner transfer, joint and sequential losses, actual supervised tokens and optimizer-step counts.
- Exact parameter/optimizer/RNG restoration evidence, original parameter hashes, all forward/backward/update/token/write costs and retained artifacts.
- Raw unsuccessful as well as successful trajectories. No best-pair, best-prefix or post-hoc metric selection.

The primary endpoint is the mean over four pairs of the final per-pair minimum total margin. The cross-arm contrast is paired-joint minus sequential-reference. The guard metric is the count among eight worlds whose final margin is below its own baseline.

## Falsification and decision

Paired joint updating is **not supported as a remedy on this fixture** if either predeclared condition fails:

1. Its mean final pair-minimum margin must be strictly greater than the sequential reference's.
2. Its guard count must be no greater than the sequential reference's.

Equality fails the primary condition. Any missing arm, restoration mismatch, original-weight change, accounting overrun or development access invalidates the comparison rather than selecting a winner.

Even if both conditions pass, the result remains a four-pair fitting diagnostic. It does not promote content keys or open development. A successor would require fresh, predeclared fitting pairs and then the complete frozen terminal gates; no such successor is selected here.

## Execution boundary and next step

No native execution is authorized by this document. Before any launch, implement the two arms with CPU regression coverage, derive and bind exact budgets and implementation closure, obtain an independent protocol review, and create a separate one-attempt/zero-retry authorization. Stop on any capacity, correctness or evidence-integrity failure.

The next smallest action is CPU implementation and review of this comparison only. No second training comparison is selected.

# Memory reliability recovery and MiniCPM5 D1 pilot

24 October 2026. **Decision: reject the learned candidate; preserve it for
diagnosis.** Benchmark/scorer integrity and persistent-state mechanics improved.
Selective recall, incremental correction quality and unrelated-answer retention
did not pass. No selected/default checkpoint changed and no RSI promotion occurred.

## What changed before training

The review of Git base `280d4dd` found a last-line shortcut in exposed M2/M3
fixtures, correction checks that rebuilt fresh states, and an M4 scorer that
trusted cached aggregates. The recovery added:

- Counterfactual episodes with identical queries and different answers; four
  target facts and two independent distractors across two shuffled pages;
  multiple reads from one state; one absent fact; an incremental correction;
  unrelated arithmetic; and held-out values crimson/indigo absent from training.
- Dataset-derived gold, exact case/arm membership, sealed-tokenizer decoding and
  aggregate recomputation. Changed gold, decoded text, correctness, token counts,
  metadata, duplicate/missing cases and changed aggregates are rejected.
- A shared SGDN-2 wrapper around the existing native writers/reducer and depth
  image format. It checks producer, scope, event uniqueness and bounds. Reload
  restores tensors/lineage without replay, then correction appends to that state.
- An independent CPU scorer that authenticates saved extension tensors, derives
  the writer/reader producer, reloads all eight depth images, checks source hashes
  and counters, decodes answers, verifies training ownership and guard policy,
  and compares the separate restart against the original tokens and state.

The fixed dataset contains six complete counterfactual families: four training,
one development and one held-out family, each with two variants. There are 96
training cases and 48 evaluation cases. First-value and last-value heuristics
score **11/48 and 9/48** fact queries; query-only choice is bounded at **24/48**
over paired worlds. These heuristic counts cover all dataset fact queries, not
just the 16 evaluation fact rows. Rendering matches the structured oracle.

The historical M4 replay verifies **434 nested answer rows**, including repeated
restart references and six external parent controls. It preserves the historical
rejection and 427-generation accounting. The four-aggregate exploit and paired
arm/metric deletion are rejected. Historical protocols/results were not edited;
their old implementation pins remain historical pins.

## Declared experiment and actual training

[Protocol v1](../../configs/memory_reliability_v1.json) was timestamped and
hash-pinned before native launch. It is **not a Git preregistration commit**;
this is an uncommitted worktree based on `280d4dd`. The
[pre-run seal](../../reports/memory_reliability_prerun_20261024.json) records the
implementation and data hashes.

| Item | Actual scope |
|---|---|
| Baseline | Sealed MiniCPM5-2B, revision `f97400052a43d642bbc6e9975e2397e3ae6a6b52` |
| Parent | `runs/m3_depth_shared_attempt02/selected_extension`; accepted ownership mechanics, failed retention |
| Training | 24 optimizer updates × 4 examples; one seeded pass over all 96 training cases |
| Trainable | 7,356,418 parameters in 24 fast writer/reader tensors; 2,516,756,480 original parameters frozen |
| Optimizer | Muon int8 momentum: 7,340,032 parameters; auxiliary FP32 AdamW: 16,386 parameters |
| Residency | FP32 parameters/gradients; no BF16-residency or FP8-training claim |
| Objective | Response-only CE; additionally KL from the disabled frozen base on 16 train-only general examples, weight 1.0 |
| Learning rate | 0.001 for both new-parameter optimizer groups; gradient norm bound 1.0 |
| Evaluation | 96 parent + 240 candidate generations; 765 generated tokens, 23,672 prompt tokens |
| Actual token bounds | Maximum source page 32, full-context prompt 135, target including EOS 4 tokens |
| Saved artifacts | One 48,948,272-byte extension, eight persistent images; no base copy or download |

Independent tensor comparison confirms that exactly the 24 declared fast-memory
tensors changed; every other extension tensor is identical to the parent.
Training gradients reach both writer and reader groups, with no original-model
gradients. First/last minibatch losses were 18.8684 and 2.1529. These are different
shuffled minibatches, not a controlled validation-loss improvement.

The first restricted-context launch stopped at macOS telemetry before creating
a native worker or loading the model. Its guard failure is preserved. The
authorized native launch then completed under the unchanged resource policy.
There was one actual training job and one separately declared restart job.

## Independently decoded results

Counts below combine development and held-out rows. Each split independently
fails the relevant quality gates. Paired rows and repeated arithmetic queries
are not independent statistical samples.

| Case type | M3 parent fast | Fine-tuned fast | Disabled base | Full-context base |
|---|---:|---:|---:|---:|
| Initial fact queries | 3/16 | **0/16** | 0/16 | 16/16 |
| Corrected fact | 4/4 | **0/4** | 0/4 | 4/4 |
| Unaffected facts after correction | 0/12 | **0/12** | 0/12 | 12/12 |
| Absent facts | 0/8 | 8/8 | 8/8 | 8/8 |
| Unrelated general answers | 0/8 | **0/8** | 8/8 | 8/8 |

The candidate's shuffled-state and last-chunk controls have the same zero
fact/correction/unaffected/general accuracies. The candidate returns
`unknown`/`Unknown` on 37/48 fast-path rows, including all 16 initial fact rows.
Correct abstention on absent facts does not establish useful memory.

The [final independent verdict](../../reports/memory_reliability_attempt02/verdict_final.json)
rejects **12 of 17 checks**. Passing checks are unknown-answer handling in both
splits, disabled-token parity, persistent correction lineage and fresh-process
token/state equality. No aggregate or lower loss overrules these quality failures.

The full-context base scores 48/48 on these same rows. This supports a failure
in the learned memory path or its interaction with the base, rather than a lack
of basic ability to answer the supplied facts. The experiment does not isolate
writer binding, query-conditioned reading and residual interference: both fast
writers/readers trained together. The parent remembering the latest correction
while losing every unaffected fact is also consistent with a recency shortcut,
not established associative correction.

## Persistence and resources

Each evaluation episode answers five initial queries from one state, saves it,
reloads it, then appends a correction and answers seven further queries. All
eight saved images authenticate scope, source hashes, exact lineage, counters
2→3, tensor bytes and the producer derived from the actual candidate extension.

A second native process loads that exact extension and the first held-out
corrected image. **7/7 token sequences and state digests match**, with no source
re-encoding, optimizer updates or new checkpoint. The resulting answers remain
wrong on six of those seven rows. This is persistence evidence, not memory quality.

| Run | Wall time | Sampled driver peak | Observed swap growth |
|---|---:|---:|---:|
| Training + all controls | 115.35 s | 11.0043 GiB | 0 |
| Fresh-process reload | 6.01 s | 10.0163 GiB | 0 |

Both native jobs passed the **50 GiB driver / 512 MiB sustained swap growth /
15% minimum host-free** guard. Training's minimum observed host-free percentage
was 76%. Sampled peaks may miss transients; MPS driver and RSS overlap.

This wrapper is a trusted synthetic diagnostic integration. It does not replace
ledger admission/revocation, prove recovery from a mid-write crash or power loss,
or erase superseded information. Latest-value correction, deletion/revocation
and neural unlearning remain distinct. Existing permitted-source rebuild rules
continue to apply to revocation.

## Next loop-agent instruction

1. Preserve this rejection, the accepted mechanical parents and all raw evidence.
   Do not select this candidate or repeat the same exposed-family recipe with
   extra steps, a changed threshold or a larger model.
2. Make durable exact-record retrieval/correction a reference using the existing
   ledger and source spans. Compare selective retrieved context with the 48/48
   full-context control. Add multi-episode/family coverage and interruption,
   duplicate-event, revocation and recovery checks before deployment claims.
3. Isolate one learned-memory failure at a time. The current writer projects
   frozen token embeddings; test whether entity/attribute/value binding survives
   writing and whether different queries read distinguishable relevant content.
   Compare contextual fact representations only under a new declared protocol.
   Do not assume the observed failure proves SGDN-2 or MiniCPM5 intrinsically
   incapable.
4. Establish a query/evidence-dependent read gate that preserves the disabled
   base on unrelated questions, including adversarial memory. Runtime evidence
   and query features are allowed; split, case role and gold labels are not.
   Audit metadata invariance. The new fixture's 16 general training examples and
   KL weight 1.0 did not suffice; that is a negative recipe result, not permission
   to tune on the exposed evaluation answers.
5. Admit another bounded D1 study only after its specific software/benchmark
   prerequisites pass, with new independent families, fixed gates and one
   discriminating change. Reliable recall, persistent correction, unaffected
   retention and noninterference must pass before any neural RSI promotion.

MiniCPM5 remains baseline. Spark BF16, MiniCPM-V, additional memory levels and
authentic personality training retain their separate qualification/data gates.

## Evidence and validation

- [Raw training/evaluation](../../runs/memory_reliability_attempt02/run_report.json),
  [candidate manifest](../../runs/memory_reliability_attempt02/candidate_extension/manifest.json),
  [independent assessment](../../reports/memory_reliability_attempt02/assessment.json).
- [Raw restart](../../runs/memory_reliability_restart01/run_report.json),
  [training guard](../../reports/memory_reliability_attempt02.guard.json),
  [restart guard](../../reports/memory_reliability_restart01.guard.json),
  [initial sandbox/telemetry failure](../../reports/memory_reliability_attempt01.guard.json).
- [Historical M4 audit](../../reports/validation_memory_reliability_m4_20261024.json),
  [token-bound audit](../../reports/validation_memory_reliability_tokens_20261024.json).
- Focused new recovery tests: **19 passed**. The combined focused run first
  found an unsupported generation keyword before native launch; it was removed
  and the new suite passed. No native budget was spent on that software error.
- Full suite: **1,280 passed, 21 subtests passed, one MPS-only skip** in 151.31 s.
  The skip is restricted-context MPS availability; the native pilot/restart
  above subsequently ran on MPS. One existing sparse-tensor warning remains.
  [JUnit](../../reports/validation_memory_reliability_20261024.xml).
- No Git commit, source-model modification, default checkpoint promotion or
  protected `runtime_pilot.py` modification/import was performed by this recovery.

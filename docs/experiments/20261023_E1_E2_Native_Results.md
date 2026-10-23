# E1/E2 connected native memory: implementation and measured results

23 October 2026 UTC / 22 October America/Chicago. Branch
`codex/d2-capacity-runtime-pilot`. Implementation/protocol commit `9601399`;
MPS autograd boundary fix `553dbf0`.

**E1 native interfaces and the E2 bounded slot-reading diagnostic pass.** The
code connects the exact ledger, eligible retrieval, contextual source writer,
explicit request/frame ownership and native ports at blocks 13/27. It is now
usable through one persistent runtime entry point. Learned reranking quality,
held-out memory quality and the E3 cognitive core remain open.

## Native evidence

The [protocol](../../configs/e1_e2_qualification_v1.json) was committed before
execution. The [completed raw report](../../reports/e1_e2_native_attempt02_20261023/run_report.json)
and [watchdog](../../reports/e1_e2_native_attempt02_20261023.guard.json) preserve
all checks, raw completion IDs/text, implementation/source hashes and timing.
Only the sealed local MiniCPM5 source was loaded. No download occurred.

| Check | Result |
|---|---|
| Initial native logits / source-prefix features | Both exactly equal: max absolute error 0 |
| Independent port 13 / 27 intervention | Finite logit changes, maxima 0.302959 / 0.116033 |
| Interleaved A → B → A requests | A unchanged; different source B changes logits |
| D1 checkpointed backward | Original tensors frozen; writer and both readers receive gradients |
| D2 checkpointed backward | All 381 original tensors receive finite gradients; no original optimizer update |
| Separate D2 source-prefix objective | Gradient reaches embeddings and block 13; does not reach block 14 or the language head |
| All known-input sense ports | Finite nonzero gradient wiring; separate from the answer objective |
| Extension-only save/reload | Nonzero branch logits exactly equal; base weights never saved |
| Excluded irrelevant / other-subject / future records | Same selected record and identical generated IDs |
| Correction and ledger restart | Old frame rejected; new record selected; answer changes from amber to blue |
| Native dense candidate path | Eligible source and producer identity verified; no semantic-quality claim |

Source spans in these native cases are **10 tokens**; answer prompts are
**37 tokens**, with a one-word answer and EOS. These are not native capacity
measurements of the full 16-unit × 128-token limit. Chunk overflow, source
boundaries and larger candidate layouts have software coverage; full-bank MPS
capacity requires a separate workload before increasing training size.

## Generated answers and causal controls

The adapter registers **4,878,850 new parameters**. The 32-update D1 answer
objective trains the **3,161,602** writer/reader parameters with a frozen original
base, using Muon 8-bit matrix momentum plus auxiliary FP32 AdamW. The separate
ranking phase trains **660,480** reranker/block-13 sensor parameters. Upper
sensors 27/41 contain **1,056,768** parameters: they receive wiring-test gradients
but no optimizer updates. The answer optimizer owns all new trainable parameters;
parameters without gradients do not receive updates. The final mean response
loss is low, but the acceptance evidence is the raw generation below.

| Source world | Expected | Initial / ports disabled | Final record slots | Exact-text fallback |
|---|---|---|---|---|
| Aster, amber record | amber | `v1.0</think>\n\nv1.0` | `amber` | `amber` |
| Aster, blue record | blue | `v1.0</think>\n\nv1.0` | `blue` | `blue` |
| Birch, amber record | amber | `Birch` | `amber` | `amber` |
| Birch, blue record | blue | `Birch` | `blue` | `blue` |

Initial and port-disabled accuracy are **0/4**. Both final slot-only and exact
fallback accuracy are **4/4**, with EOS after the single answer token. The query
is identical within each entity pair and never contains the source label.
Counterfactual record swaps therefore identify a causal memory contribution in
both pairs. The host returns raw decoded model tokens; it does not substitute
an answer from the database.

The exact fallback fetches the selected source span under the evidence pin and
uses a fresh text prompt with memory ports disabled. It is an access fallback,
not evidence that compression improves accuracy. The disabled-port outputs
match the unmodified initial model's behavior.

After a correction replaces the Aster amber record with the blue record, the
old frame fails verification, the reopened ledger selects only
`aster_corrected`, and both access paths generate `blue`. Slot cache identity
and extension reload were verified in process. The separately declared
[fresh-process runtime smoke](../../configs/e1_e2_restart_smoke_v1.json) also
**passed**: the production CLI reloaded the extension and ledger, selected
`aster_corrected`, reused one persisted latent (zero source-feature encodes),
and generated `blue</s>` without full source text in the query. It used a
73-token runtime prompt, took **4.89 seconds** including load, and peaked at
**10.02 GiB** sampled driver memory. There were no updates, downloads or saves.
See the [restart acceptance](../../reports/e1_e2_runtime_restart_20261023/acceptance.json)
and [raw session report](../../reports/e1_e2_runtime_restart_20261023/session_report.json).

These four worlds are **exposed synthetic training/development fixtures**. The
correction also uses a seen entity/value combination. This demonstrates connected
access and causality; it does not measure novel-fact retention, reasoning,
identity fidelity, long-term learning or generalization.

## Reranker limitation

Four separate ranking updates train only the reranker and block-13 query sensor
on two eligible records. Top-1 accuracy remains **1/2**: before training both
queries prefer Aster, afterward both prefer Birch. Loss moves from 0.692673 to
0.690995. The objective and gradients work, but query-conditioned discrimination
is not learned sufficiently in this budget. The runtime therefore leaves
reranking opt-in. This is an open quality gate, not a successful retrieval result.

Answer training does not backpropagate through hard top-k. Future work needs a
source-separated ranking dataset, relevant distractors, better query alignment
and a declared budget. Increasing these four updates until the tiny pair passes
would not establish retrieval quality. Likewise, the irrelevant-record control
here tests host candidate exclusion; robustness when irrelevant records enter
the neural bank remains unmeasured.

## Memory, time and disk

| Completed attempt 2 measurement | Value |
|---|---:|
| Total profiled wall time, including source load/checks and training | 54.26 seconds |
| Sampled peak MPS tensor allocation | 18.84 GiB |
| Sampled peak MPS driver allocation | 21.11 GiB |
| Process lifetime peak RSS | 14.49 GiB |
| Maximum observed system swap growth | 0 MiB |
| Minimum observed system free-memory percentage | 54% |
| Final extension file | 19,519,840 bytes / 18.62 MiB |
| New original/base checkpoint copies | 0 |
| Successful run artifacts before final report | 19,836,939 bytes / 18.92 MiB |
| Observed filesystem free-space decrease during attempt | 22,183,936 bytes / 21.16 MiB |

MPS and RSS overlap in unified memory and must not be added. Peaks are sampled;
system disk/swap changes can include other processes. Source weights remain the
single original template at `checkpoints/minicpm5_source`. The selected extension
is `runs/e1_e2_native_attempt02_20261023/selected_extension`; it is an inference
weight export with source references, not a resumable optimizer checkpoint.
The [extension manifest](../../reports/e1_e2_native_attempt02_20261023/extension_manifest.json)
binds its exact inventory and checksum. All binary payloads remain ignored. All three new run directories, including
both attempts and fresh-process reports, total **20,460,649 bytes / 19.51 MiB**;
only one new extension weight file exists. See the
[preservation/storage receipt](../../reports/e1_e2_preservation_after_20261023.json).

## Preserved failure and total work

Attempt 1 completed all 32 alignment updates and generated 4/4 correct slot
answers, then failed on the first reranker backward. Its loss hook called SQLite
from an MPS autograd worker thread. CPU tests had run that hook on their calling
thread. The fix removes host storage from autograd hooks; model/frame tensor
checks stay in the hook, while source scope/epoch checks run on the host before
and after backward. A new regression test forbids host storage access in the
ranking hook. See [failure analysis](../../reports/e1_e2_native_attempt01_20261023/analysis.json)
and [raw failure](../../reports/e1_e2_native_attempt01_20261023/failure_report.json).

The failed process had not exported its learned weights, so attempt 2 repeats
the unchanged protocol from the sealed source. Total completed optimization
work across attempts is **64 D1 alignment updates plus four ranking updates**;
the selected extension contains the second attempt's 32+4 updates. There are
also the declared backward-only wiring probes. No D2 original-weight update or
full-base checkpoint is hidden in this accounting. Attempt 1 took 52.45 seconds;
attempt 2 took 54.26 seconds. All failures and prior experiments are preserved.

## Software and next work

The complete pre-native suite passed **922 tests, one restricted-context MPS
skip and 21 subtests**. After the MPS-specific fix, the affected adapter/session/
CLI suite passed **60 tests**, including the new storage-free-hook regression.
The native run then exercised that fixed path successfully. See
[software evidence](../../reports/e1_e2_software_validation_20261023.json).

An independent read-only agent review checked raw answers, scope, hashes and
storage claims; its parameter-count wording correction is applied above. See
[review receipt](../../reports/e1_e2_independent_review_20261023.json).

Proceed to E3's shared recurrent core and exact task/action journal using these
interfaces. Keep E2 ranking, full-bank capacity, multi-record interference and
novel-source evaluations visible as separate work. No broad corpus campaign,
large-model port, ANE execution, VLM integration or self-improvement promotion
is established by this result.

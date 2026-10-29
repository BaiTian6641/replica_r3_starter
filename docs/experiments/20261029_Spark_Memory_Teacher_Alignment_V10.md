# Frozen-teacher intrinsic-memory comparison v10

This separately versioned experiment follows the completed v9 direct-CE control:
8/8 frozen competence, final fitting 6/8 after the fixed 128 updates, unopened
development, exact fresh-process logits and one actual continuation update.
The reference and all earlier results remain unchanged. V10 is not a continuation
of v9's extension weights; both start from the same declared initialization.

The source is the existing official `XHToken/Spark-X2.6-4B` BF16 artifact at
revision `66e3ae7bee54e0a523d692cb8ef841f35409807b`, in `.venv-spark-bf16`.
The FP8 source lineage, official FP8 execution and the FP8-derived BF16 artifact
retain separate identities. No source patch, dependency substitution, download,
duplicate backbone or full-backbone export is part of this experiment.

## Intervention and boundaries

The method reference is the response-aligned distillation objective in
[MemT-1.0, sections 3.3 and E.2.3](https://arxiv.org/html/2610.26825v2).
Teacher and student use the same observed response prefix. A frozen full-context
teacher supplies top-64 targets, normalized on that support; the student remains
normalized over the complete vocabulary. Alignment follows response-token index
because their prompt lengths differ. This experiment tests that objective in
Replica's existing memory implementation; it does not adopt the paper's memory
architecture or inherit its capability claims.

V10 uses temperature 1, KL weight 1 and CE weight 0. Complete-response CE is still
reported as a diagnostic, including separate answer-word and terminal losses.
Losses cover each response token and exactly one bound EOS. Prompt and padding
are excluded. The canonical response IDs are identical in the two paths; no
additional causal shift or absolute-position alignment is permitted.

The data, 0.0001 learning rate, seed 17, 128+1 student updates, fixed 32/64/96/128
probes, batch size, optimizer, source, template, scorer, generation limits and
memory mechanism match v9. The changes are the fitting objective, an explicit
teacher stage and the necessary execution/output/budget identities. Development
from the v8/v9 corpus remains unopened until the fixed final fitting gate passes.
These data cover one synthetic rule family and seed; this is a feasibility
comparison, not independent-family generalization evidence.

Only original frozen weights serve as teacher, with all extensions disabled.
There is one model object per worker and no teacher model copy. The producer
continues to consume eligible canonical source spans, not teacher answers,
queries or label-bearing cache state. Every active reader consumes the same
snapshot using block-input queries and post-block residuals. Pre-MLP fusion is
not introduced. Original attention, positions, gated attention, GELU MLP and the
tied output path are preserved. The external spine still connects specialists,
sensors and actors.

## Ordered gates and target lifecycle

1. Validate the exact environment, source files, model/configuration executable
   closure, protocol, data hashes, predecessor and output before loading weights.
2. Run the original frozen complete-context calibration: 8/8 correct, zero false
   recall and no truncation. Failure closes teacher preparation and fitting.
3. Open fitting labels. Check disabled/zero-output parity and admit the same
   eight source records under current scope, cutoff and authority.
4. Generate complete-context answers on all eight fitting worlds using the
   frozen teacher. Require 8/8 correct, zero false recall, no truncation and
   termination. Preserve every raw answer. A teacher failure closes fitting.
5. Make eight additional teacher-forced forwards, aligning response targets and
   recording retained probability mass. Store one immutable, at-most-2-MiB cache.
6. Run all 128 planned student updates and all four probe prefixes. Only the
   final prefix can qualify: 8/8 complete first-token-correct terminated responses,
   exact fitting/generation prefix logits and better recall than wrong/zero/absent.
   No best-prefix selection, early stop or adaptive development tuning is allowed.
7. Freeze and checkpoint the extension/optimizer, and evict teacher targets/responses
   from the engine before any development access. The utility criteria remain contextual recall at least
   6/8, zero false recall, at most one answer behind exact text, and strictly better
   than wrong, shuffled, zero and absent values. Corrections and revocations retain
   their separate storage-validity and generated-answer measurements.
8. A fresh process restores exact logits, the extension/optimizer and
   the teacher-cache identity, then performs one actual KL continuation update.
   Full original-parameter hashes are checked in both processes. This is not a
   claim of equality with an uninterrupted next update.

Teacher tensors deliberately detach before serialization. Cache identity includes
the full original-parameter digest, fitting-data digest, world/source text and
question/response digest, tokenizer/template/stop objective, teacher and student
input digests, their separate prediction positions, support IDs and probabilities.
Missing, oversized, substituted, corrupted, unqualified or mismatched targets
refuse. Target validation precedes the student loss. Current store authority is
also revalidated: an intact teacher cache cannot revive corrected/revoked evidence.

Student contextual values remain live tensors. Neither NumPy, hashing nor durable
serialization lies on their loss-to-producer gradient path. Zero-initialized
reader outputs initially suppress upstream producer gradients; after they begin
learning, the producer and every active reader must receive finite gradients.
One NativeFusedAdamW owner holds each intended producer/order/reader parameter
exactly once. Original weights and detached teacher targets have no optimizer owner.

## Work and physical resource budgets

Each 2/9/36-site arm has its own packet. No density arm is authorized by a useful
result from another protocol or by a parameter-count estimate. Matching useful
two-site evidence, numerical continuation and healthy resource evidence are
required first.

| Quantity | Maximum per arm |
|---|---:|
| Model forwards | 2,469 |
| Student backward calls / updates | 129 / 129 |
| Generated tokens, including stops | 2,072 |
| Padded input positions | 632,064 |
| Sequential original-model loads | 2 |
| Source encodes / retrievals | 245 / 51 |
| Declared store/checkpoint/teacher writes | 112 |
| Stage wall allocation / outer wall | 7,080 / 7,200 s |
| Artifact cap, 2 / 9 / 36 sites | 160 / 416 / 1,056 MiB |
| Teacher/backbone updates / P updates | 0 / 0 |

The added teacher stage reserves 72 forwards: up to 64 generation steps plus
eight target forwards. It adds 64 generated tokens, 18,432 padded positions and
nine writes (eight qualification receipts and one cache publication). Its 120 s
comes from reducing calibration's time allocation from 600 to 480 s; calibration
criteria and token limits are unchanged. The 2-MiB cache reservation comes from
the existing 32-MiB audit/store envelope, leaving 30 MiB there. Total wall and
artifact caps remain unchanged. Reused targets require validation and bounded
read/hash work, but no additional teacher forward or backward during updates.

Keep one exclusive native worker, 50 GiB sampled driver memory, 512 MiB sustained
swap growth and at least 15% host free. Driver and RSS overlap. The inherited
process-group supervisor covers preflight, main and restart; telemetry and stage
deadlines remain enforced. No retry, fallback, promotion or capacity expansion
is implicit. Raw answers, failed work, audit overhead, teacher-cache cost, total
tokens, latency and physical-memory samples remain part of the result.

## CPU engineering and authorization

The objective tests check response/EOS alignment, full-vocabulary normalization,
detached teacher targets, source/identity rejection and serialization parity.
Reader tests cover live gradients at 2/9/36 sites, skipped disabled branches,
correct/wrong/shuffled/zero/absent controls, correction invalidation, cache
tampering, complete checkpoint/optimizer ownership and actual fresh continuation.
Native-controller fixtures verify that failed baseline or teacher competence
opens neither fitting nor development. Source tests use synthetic executable
trees; real source admission remains an explicit integration check.

The separate CPU workflow uses a tiny random BF16 backbone and explicitly
synthetic teacher qualification. Its actual target forwards, gradients, durable
round trips and fresh-process updates are engineering evidence. It does not
verify pretrained teacher answers or useful recall. Results and native status
belong to the [v10 runbook](../../reports/intrinsic_memory_alignment_v10/README.md).

The user's standing direct instruction authorizes bounded testing without another
approval question. Record it in the exact new packet receipt. All consumed v5–v9
outputs remain consumed with zero retries. Persistent learning, formation, native
multimodal work, nested adaptation and RSI remain behind their existing gates.
The Replica sequence remains useful memory, attributable interaction formation,
verified persistent learning, retention/correction/rollback, then bounded cognitive
and approved-identity integration with distinct E/S/P/phi/C/I/K ownership.

## Completed native outcome

Execution at `7c44f417` used the existing official **BF16** artifact and the
qualified environment, under packet SHA-256
`1369228284d68d91354f05833ad4dffb71deee99fb14ef24b977acf386bb48d9`.
The user’s standing testing instruction was recorded in an exact receipt before
launch. This two-site output is consumed with zero retries. Original FP8 lineage,
the previous FP8-derived BF16 results, and all v5–v9 attempts remain distinct.
No source patch, dependency substitution, download or base export occurred.

Both frozen calibration and the fitting-teacher qualification pass **8/8**,
with zero false recall, zero truncation and complete termination. The teacher
used 24 forwards (16 generated tokens and eight target forwards), 6,144 padded
input positions and 8.072361 s. The detached cache is
72,427 bytes; student updates used its bound distributions without
additional teacher forwards. Gradients reached the producer and both readers
after zero output initialization began learning. One NativeFusedAdamW owned each
intended trainable parameter. Original weights and teacher targets were frozen.

The predeclared fitting curve was:

| Updates | Contextual /8 | Exact /8 | Wrong /8 | Shuffled /8 | Zero /8 | Absent /8 | Both-correct pairs /4 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 3 | 8 | 3 | 3 | 2 | 2 | 0 |
| 64 | 6 | 8 | 2 | 6 | 2 | 2 | 2 |
| 96 | 5 | 8 | 3 | 5 | 2 | 2 | 1 |
| 128 | 8 | 8 | 0 | 8 | 2 | 2 | 4 |

Only update 128 was eligible for the fitting gate. It passed 8/8 and all four
same-question counterfactual pairs. No earlier prefix or checkpoint was selected.
The matched direct-CE v9 reference ended at 6/8; this one-seed fitting comparison
does not establish statistical superiority or broader generalization.

Fitting froze and the teacher cache was evicted before development opened.
Development contextual/exact/wrong/shuffled/zero/absent recall was
**5/8, 8/8, 0/8, 5/8, 1/8 and 1/8**; false recall was **0, 0, 1, 0, 0 and 0**.
Every contextual response terminated without evidence truncation. The three
contextual errors were abstentions on supported blue/gold answers. This is
partial source-sensitive transfer, but it fails the fixed >=6/8 threshold,
maximum one-answer gap to exact text, and strict shuffled-control comparison.
**Useful recall is not qualified. Native 9/36, formation, P learning and promotion
remain closed.**

Correction rejects the stale snapshot and revoked retrieval. The corrected
contextual answer is `unknown`, while matched exact text correctly answers
`blue`; after revocation, the absent path correctly answers unknown. Storage
validity passes, but corrected-answer utility does not. The expected rejected
revocation read is preserved as a failed retrieval event in the work ledger;
it is not an interrupted native run or a reason to weaken production validation.

The fresh process restored the full extension, optimizer ownership and detached
teacher cache, reproduced exact logits and performed one actual continuation
update (129 total student updates, zero P updates). Full original-parameter
SHA-256 stayed `39ea150a4d2ed39184f2a83fbd474b67644e4d38f4039177c72cedd069ef4747`.
Uninterrupted-next-update equivalence was not measured. The supervisor confirmed
process-group exit with no termination signal or resource stop.

| Actual resource/work | Measurement |
|---|---:|
| End-to-end supervisor wall | 617.853465 s |
| Peak sampled driver memory | 11,261,542,400 bytes (10.488129 GiB) |
| Peak process RSS | 8,700,542,976 bytes |
| Peak swap growth / minimum host free | 0 bytes / 70% |
| Output directory bytes after exit | 49,944,544 |
| Model forwards / student backward-updates | 922 / 129 |
| Generated tokens, including stops / padded input positions | 525 / 236,032 |
| Sequential model loads | 2 |
| Source encodes / retrievals / declared writes | 245 / 51 / 112 |
| Audit serialized bytes / source-file hash bytes | 3,084,104,699 / 24,716,988,741 |

Driver and RSS overlap and must not be summed. These are sampled peaks, not hard
real-time guarantees. Swap growth is relative to launch baseline. Serialized
write and hash-byte counters are not physical SSD or bus traffic. The full stage
breakdown and failed/positive raw trajectories remain in the attempt directory.

The new clean CPU checks at `ad58c14b` pass **502/502**, zero skips, in 226.228 s
(226.823251 s wall; 794,263,552-byte child peak RSS). The 2/9/36 integrated CPU
workflow passes in 14.439167 s with 615,071,744-byte parent peak RSS and
105,446,031 bytes across arm directories. Its synthetic teacher qualification and
tiny random backbone are explicitly engineering evidence. Earlier 444/393/3,371
receipts retain their original scope; the 3,371-test scope was not rerun here.

Posthoc fitting-only inspection found distinct counterfactual encoder values:
v10 relative RMS differences are 0.252–0.366. Ordered-position embedding RMS is
0.0203 versus contextual slot RMS 0.6089. These observations neither prove an
encoding collapse nor identify the cause of the native transfer errors. The
teacher distributions assign approximately 0.826–0.978 probability to the
canonical first answer token and >0.9999 to EOS. Reproducible analysis scripts
read only completed receipts and fitting data; they load no native backbone and
perform no update.

The v8 development split shared by v9/v10 is now **exposed by v10**. Its
[exposure receipt](../../reports/intrinsic_memory_alignment_v10/development_exposure.json) prevents a successor
from calling it fresh. Historical v8/v9 closed-label receipts remain correct for
those attempts. No sealed-final labels were opened. A new native comparison needs
fresh, predeclared development and one separately bound intervention with fixed
gates and full costs. Broader training coverage and order-sensitive readout are
questions for that next protocol, not implemented fixes or an admitted launch.
Do not repeat a consumed packet, choose a better observed fitting prefix, relax
acceptance, or bypass useful recall by allocating more readers or P capacity.
The standing user authorization continues to cover bounded tests; another
permission question is not needed.

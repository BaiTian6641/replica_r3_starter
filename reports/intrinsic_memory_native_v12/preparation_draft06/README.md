# Intrinsic memory v12 native preparation

Status: **prepared, not launched, not reviewed, not sealed.** No original weight
was loaded, no development label was parsed, no authorization receipt exists and
no native attempt was consumed. This directory holds preparation and review
material for the separately versioned v12 source-feature successor to the
consumed v11 two-site attempt.

Current preparation (post-fresh-split, after repair cycles 1–3 and the X0/X2/X3/X9
repairs): **80 focused tests passed** in `tests/test_intrinsic_budget_v12.py`,
`tests/test_intrinsic_coverage_v12.py` and `tests/test_intrinsic_native_v12.py`,
run by me against these bytes. Envelope: stage wall 6845 s inside 7200 s, 4029
prompt-tokenizer passes, 42,647,926,441 declared file-read bytes (including the
450,061,586-byte environment closure in admission, load and restart), 2049
declared writes, restart 150 s / 67 passes.

**X0 — fresh development split.** The v11 development split is *retired* for
this successor: the re-review scout read
`data/intrinsic_memory_coverage_v11_20261029/development.json`, so it can no
longer support a fresh utility claim. Calibration and fitting keep the exposed
v11 rows for a matched comparison. Development is new
(`data/intrinsic_memory_coverage_v12_20261029/development.json`, sixteen cases in
two groups of eight, new entities `Thalric…Halvra`, world ids
`v12_development_00…15`, roots `v12_development_decision_00…07`), generated
deterministically from the sealed structured oracle with no v11 entity, root or
path overlap. Labels were generated, never inspected or used for selection, and
the file is pinned before any observation. The current receipt
`development_exposure_v12.json` records `development_opened: false`, the
retired split and the reading incident; the historical v11 exposure receipt is
byte-unchanged. Admission re-validates the role binding, the fresh identity and
that the retired split is not reused, without parsing development labels.

**X2 — torn ledger tail.** Every mutation (`enter`, `reserve`, `finish`,
`leave`) and every non-operator read (`snapshot`, `deadline_state`, `audit_io`)
now refuses on a torn tail instead of fusing a new record onto the partial line.
Recovery is explicit: `recover_interrupted(accept_torn_tail=True)` truncates the
fragment, appends a `torn` record carrying its length and digest, closes
interrupted reservations as failed while leaving them charged, and only then
allows further work. Three tests cover mutation refusal, read refusal and
operator recovery.

**X3 — charged work.** Source preflight/revalidation hashing, environment
closure hashing, model loads, per-fit receipts, probe/checkpoint/development/
correction/restart receipts and stage source-cost receipts are each reserved
before the operation, with the actual observation recorded and no refund. New
whole-stage tests consume exactly the declared caps for `fit_128` (128 receipts
+ 1 source-cost = 129), the full `restart` reservation set, and the admission
hashing envelope (one extra byte is refused before dispatch).

**X9 — attempt consumption and environment closure.** Consumption no longer
relies on the deletable output directory: `launch` appends a durable
append-only marker (`runs/intrinsic_memory_native_v12/attempts.jsonl`) bound to
its packet hash and output root *before* creating the output tree; supervised
workers require that marker; a torn marker blocks the attempt. The environment
closure now binds the executable dependency set by actual hashes — the resolved
interpreter, the declared module origins, the loaded native extensions, all
twelve `torch/lib` files, all 32 wheel `RECORD` files and the sealed receipt's
critical files — with exact set comparison so an added or removed library is a
failure. Size is pinned at 67 files / 450,061,586 bytes and re-derived from the
bound receipt.

Current packet SHA-256 is
`48963bdc1996526d7512cb6e753901d3562b08610cf67894906f0ec95b988c9e`; protocol
SHA-256 `16c7b2f3ff072249e9a370be6e906ac93d8cfced6127122dfa6b6337e09203e7`;
implementation receipt SHA-256
`c683db4881346ff0d922fc1744557f00a1328d462550b4facea5239362185d2c`;
environment closure SHA-256
`d521db9340e3638c481ede19a7ed670949ed9440167739b426b815276d6fbc68`;
development exposure receipt SHA-256
`79e8b3abf656cd41a8c5681ed45a0ef7be9b48fc610831e905c06c16527ed5d3`.

The earlier coordinator preflight (57 focused tests, environment-only, and
`--review` over 62 implementation files, two source-execution files and
8,238,996,247 source bytes) applies to the superseded packet
`3a2a7af1d08118df902e1dd20f5b3ca74b1c651005a2e0bf7f9eda7868198cc2`. Its exact
commands and results remain in `preflight_verification.json` and the initial
18-failure run in `verification_attempt01.log`. **Environment-only and `--review`
have not been run against the current packet**; `--environment-only`, `--review`,
the full suite and the independent reproduction are the coordinator's next gates,
and independent review is still required before authorization or launch. Note
that `--review`/`--launch` now also verify the 450,061,586-byte environment
closure and, for launch, the durable attempt marker.

Five earlier unlaunched preparations are preserved verbatim, unmodified and
clearly non-canonical, each with a computed-hash status record:
`preparation_draft01/` binds pre-repair bytes, `preparation_draft02/` carries the
N1–N3 repairs made before the producer commit, and `preparation_draft03/` carries
the retired v11-development reuse contract before the fresh split and the
X-series repairs. `preparation_draft04/` preserves the working-tree packet made while producer
cycle 4 was uncommitted, and `preparation_draft05/` preserves the packet made
while the consumer repairs were uncommitted. Both drafts' config, split
manifest, fresh development corpus, packet, implementation receipt, environment
closure and exposure receipt are byte-identical to the current canonical ones;
only their `preparation.json` differs (measured hashing wall time). No attempt or
historical evidence was moved or deleted.

Provenance: the v12 producer was committed at `6dede67e`, repaired for the
independent review at `6ba07e1b`, and repaired again for the re-review at
`826aac6b` (`fix: charge cold failures and prove release parity`; charged cold
failure payload reads, typed failure preserved through best-effort eviction,
with 79 coordinator checks and all-layout release parity). The matched
density/measured-cost controls are at `3cd683c3`; the density modules are a
separate track and are not part of this experiment's executable closure.

The five producer modules in this closure are **committed** blobs: every
producer hash in `implementation_v12.json` equals its `HEAD` blob content at
`826aac6b`. The seven consumer files of the X0/X2/X3/X9 repairs
(`intrinsic_memory_{budget,coverage,diagnostic,execution,native}_v12.py` and
both runner/preparation scripts) are committed at `257c722e` (`fix: close native
recovery and fresh-split gates`). **All 62 closure entries now equal their `HEAD`
blob content**, and the regenerated packet is byte-identical to the one the
coordinator gates passed — only `preparation.json` differs, in its measured
hashing wall time. `--review` recomputes the closure statically and refuses any
difference, so a future `src/` change fails closed until re-preparation.

Bound after regeneration: the content-attestation change (the adapter contract
includes the exact base content digest, supplied as the already-verified full
original parameter digest), the complete-representation release (compressor,
order, reader and feature contract), and the checkpoint manifest's full
release/alias/compressor-only releases with outer/inner cross-checks. All five
producer modules and the local executable closure are pinned by hash in
`implementation_v12.json`; `--review` recomputes the closure statically and
refuses any difference, so a later `src/` change fails closed until
re-preparation.

Repair cycle 2 (independent review `f98247f8`, this directory) fixed three
native findings and refreshed the envelope:

* **N1 — restart prompt-tokenization cap.** Restart restores 32 teacher targets;
  each revalidation performs a student and an exact-text encoding (64), and the
  canary plus the continuation update add three more, so the stage now reserves
  **67** passes instead of 4. The cap is derived in code
  (`RESTART_PROMPT_PASSES = 2*FIT_CASES + 3`), `validate_protocol` refuses any
  protocol whose restart cap is below the required passes, and a real-ledger
  consumption test proves the 67 reservations are usable and the 68th is
  refused before dispatch.
* **N2 — truncation.** Generation that exhausts the token budget without the
  bound stop token now reports `terminated: false, truncated: true`, and
  baseline competence additionally requires `terminated` for every calibration
  answer.
* **N3 — tokenizer identity.** `BoundSourceTokenizer` no longer compares a
  declaration with itself: it re-hashes the six tokenizer files in the verified
  source directory, reads vocabulary size, special-token ids and the
  fast-tokenizer property from the loaded object, requires the tokenizer's
  `name_or_path` to be that directory, and only then compares the derived
  mapping with the reviewed declaration. A substituted file byte, vocabulary
  size, property or origin is refused before any source work. The independent
  full-source file and parameter hashes remain the primary authority check.

Envelope after cycle 2: stage wall **6845 s** (restart raised from 140 s to
150 s with explicit arithmetic) inside the 7200 s ceiling; 4029 prompt-tokenizer
passes; 41,297,741,683 declared file-read bytes (admission hashing,
per-worker pre-load hashing plus weight loads, and the restart restores);
writes unchanged at 2049.

Remaining risks recorded, not claimed as passes:

* **Attempt consumption is durable now, but recovery is operator work.** A
  marker record is appended before the output tree exists, so deleting the
  directory cannot restore an attempt; a crash between the marker and the first
  receipt still consumes the attempt with zero retries and requires a new
  preparation revision and receipt. Not behaviourally exercised natively.
* **Torn tails block rather than repair themselves.** Mutations and reads refuse
  until `recover_interrupted(accept_torn_tail=True)`; the fragment's length and
  digest are recorded but its content is discarded, so work it might have
  described is unknown and stays uncharged by construction.
* **Ledger read/write window.** Readers take the same lock shared and writers
  hold it exclusively across the append, so a snapshot cannot observe a partial
  line; the size-keyed cache is only consulted under that lock.
* **Read-byte accounting is declared, not measured.** Source hashing, closure
  hashing, weight loads, restores and receipt writes are reserved and their
  actuals recorded; physical bandwidth and page-cache reuse are not measured.
* **Environment closure breadth.** The closure binds the resolved interpreter,
  module origins, native extensions, every `torch/lib` file, every wheel RECORD
  and the sealed receipt's critical files, with exact set comparison. It does
  not hash every file of every wheel, and it cannot see a library loaded from
  outside that set; `--environment-only` still runs first.
* **Restart wall headroom.** 150 s covers model load, 8.2 GB hashing, the
  closure pass, restore, 67 encodings, two forwards, one update and two
  parameter digests; it is an allocation, not a measurement.
* **Fresh-split breadth.** Sixteen cases in two groups of eight from two rule
  families are a feasibility test, not broad generalization; false-recall
  coverage is limited to those groups.
* Producer-side findings (V1–V5, X5, X8, X12) belong to the producer track and
  are outside this packet's files.

Re-preparation after any further `src/` change (remove only the preparation
outputs, never an attempt directory, then rerun):

```
rm -f configs/spark_intrinsic_coverage_v12.json \
      reports/intrinsic_memory_native_v12/implementation_v12.json \
      reports/intrinsic_memory_native_v12/native_packet_two_v12.json \
      reports/intrinsic_memory_native_v12/preparation.json \
      reports/intrinsic_memory_native_v12/environment_closure_v12.json \
      reports/intrinsic_memory_native_v12/development_exposure_v12.json
rm -rf data/intrinsic_memory_coverage_v12_20261029
.venv/bin/python scripts/prepare_spark_intrinsic_coverage_v12.py
```

`runs/intrinsic_memory_native_v12/attempts.jsonl` must never be deleted to make a
re-preparation succeed: it is the durable record that an attempt was consumed,
and preparation refuses while it exists.

This packet makes no readiness claim.

Mechanism under test: frozen canonical-span Spark decoder hidden states,
followed by a live trainable compressor into four ordered width-256 slots per
admitted span. Sites 3/7, block-input query to post-block residual, the teacher
objective, the optimizer, the exposure, the template, the scorer and every group
gate stay matched to v11. The hypothesis is about source representation only.

## Files

| Path | Contents |
|---|---|
| `configs/spark_intrinsic_coverage_v12.json` | Complete v12 protocol, including the re-derived envelope and the declared difference list |
| `data/intrinsic_memory_coverage_v12_20261029/split_manifest.json` | New manifest reaching the same calibration/fitting/development rows as v11 with an explicit exposure contract |
| `reports/intrinsic_memory_native_v12/implementation_v12.json` | Statically resolved local executable import closure (62 files) |
| `reports/intrinsic_memory_native_v12/native_packet_two_v12.json` | The single two-site packet; dense arms stay gated and unpacketized |
| `reports/intrinsic_memory_native_v12/preparation.json` | What was and was not done, the derivation, the intended commands |
| `reports/intrinsic_memory_native_v12/preparation_draft01/` | Preserved pre-repair preparation (README, config, manifest, closure, packet, receipt) with its own status record; non-canonical, never launched |
| `reports/intrinsic_memory_native_v12/preparation_draft02/` | Preserved N1–N3 preparation made before the producer commit; its config, packet and closure matched that moment; non-canonical, never launched |
| `reports/intrinsic_memory_native_v12/preparation_draft03/` | Preserved packet with the retired v11-development reuse contract, before the fresh split and X-series repairs; non-canonical, never launched |
| `reports/intrinsic_memory_native_v12/preparation_draft04/` | Preserved working-tree packet made while producer cycle 4 was uncommitted; all artifacts except `preparation.json` are byte-identical to the current ones; non-canonical, never launched |
| `reports/intrinsic_memory_native_v12/preparation_draft05/` | Preserved packet made while the consumer repairs were uncommitted; all artifacts except `preparation.json` are byte-identical to the current ones; non-canonical, never launched |
| `reports/intrinsic_memory_native_v12/environment_closure_v12.json` | Executable dependency closure: resolved interpreter, module origins, native extensions, all `torch/lib` files, all wheel RECORDs and critical files, by actual hash |
| `reports/intrinsic_memory_native_v12/development_exposure_v12.json` | Current exposure receipt: fresh split unopened, retired v11 split, reading incident recorded; historical receipt unedited |

Produced by:

```
.venv/bin/python scripts/prepare_spark_intrinsic_coverage_v12.py
```

The script refuses to run if any of its outputs already exist. It does not
create `authorization_two_v12.json`.

## Data and exposure

The v12 study compares against v11, so it uses the *same rows*:

* calibration and fitting are already exposed (they were executed and measured
  in v11). Reusing them is lawful for a matched comparison, and no tuning may
  escape the fixed groups;
* development is reused **only** because
  `reports/intrinsic_memory_coverage_v11/development_exposure.json` still
  records `development_opened: false` and `fresh_for_successor: true` with the
  v11 development digest. Admission re-reads that receipt and refuses if it
  changed. Preparation never parses the development file: it hashes it.

A new v12 manifest is required so this reuse is explicit rather than implied.

## Work accounting

`src/replica_lab/intrinsic_memory_budget_v12.py` is a new append-only durable
journal. Its vocabulary is the v5 counters plus source-feature forwards,
canonical-source tokenizer passes and bytes, prompt tokenizer passes, cache
misses/evictions/recomputations, artifact read bytes and restart restores.

Reservation semantics:

* every reservation is written and fsynced before the tokenizer or frozen
  forward it covers, through the producer's own required accounting interface;
* totals are counted over all reservations, never over successes; a failed or
  interrupted reservation stays charged;
* the plan and absolute deadline are immutable once written; the deadline is
  persisted in both monotonic and wall clock and the stricter remaining time
  wins, so a restart cannot extend the budget;
* a torn final line is fail-closed. `recover_interrupted()` records terminated
  work as charged and refuses to guess;
* appended bytes are the physical append requests. SSD write amplification is
  **not** measured and is not claimed to be lower than v11. Lower audit cost is
  not better learning.

v5 rewrote one JSON document per event (37,101,387,705 logical bytes over 14,093
calls in the completed v11 attempt). The v12 journal appends one line per event,
which bounds the audit cost per event instead of growing it quadratically. That
is an engineering change, declared as such, with fail-closed recovery tests.

## Budget derivation (two-site arm, worst case)

Source-span unit: one cache-missing canonical span reserves
`tokenizer_calls=1`, `source_encodes=1`, `source_token_bytes=256`,
`input_tokens=256` for the tokenizer, then `forwards=1`,
`source_feature_forwards=1`, `cache_misses/evictions/recomputations=1`,
`input_tokens=256` for the frozen source forward. Each backbone decode step
reserves `forwards=1`, `input_tokens=256` (fixed right-padded width).

| Stage | Wall (s) | Forwards | Source spans (tok / fwd) | Backward | Generated | Retrievals | Writes |
|---|---|---|---|---|---|---|---|
| admission | 30 | 0 | 0 / 0 | 0 | 0 | 0 | 5 |
| load | 180 | 0 | 0 / 0 | 0 | 0 | 0 | 0 |
| baseline | 45 | 64 | 0 / 0 | 0 | 64 | 0 | 9 |
| numerical | 30 | 3 | 1 / 1 | 0 | 0 | 0 | 2 |
| prepare | 60 | 32 | 32 / 32 | 0 | 0 | 32 | 193 |
| teacher | 90 | 288 | 0 / 0 | 0 | 256 | 0 | 34 |
| fit_128..512 (each) | 500 | 256 | 128 / 128 | 128 | 0 | 0 | 129 |
| fit_probes_128..512 (each) | 900 | 1888 | 128 / 96 | 0 | 1536 | 32 | 291 |
| checkpoint | 60 | 2 | 1 / 1 | 0 | 0 | 0 | 7 |
| development | 540 | 784 | 16 / 16 | 0 | 768 | 16 | 98 |
| correction | 60 | 25 | 2 / 1 | 0 | 24 | 2 | 14 |
| restart | 150 | 4 | 2 / 2 | 1 | 0 | 1 | 7 |

Totals: stage wall **6845 s** inside the unchanged **7200 s** outer envelope
(355 s outer headroom, never a stage allowance); 9778 forwards; 949 source
feature forwards; 1078 canonical-source tokenizer passes; 4029 prompt tokenizer
passes; 513 backwards; 513 updates (512 fitting + 1 fresh-process
continuation); 7256 generated tokens; 179 retrievals; 2049 declared writes;
**42,647,926,441** declared file-read bytes; 194 MiB artifacts (96 MiB envelope
+ 2 MiB teacher cache + 96 MiB checkpoint); 2 model loads.

Restart alone reserves 150 s, 67 prompt-tokenizer passes (64 teacher-target
revalidations + canary + two update encodings), 17,030,814,528 read bytes (one
source-hashing pass plus the weight load plus one environment-closure pass plus
the two restores), and 2 source spans recomputed from a cleared cache.

Read-byte accounting: 8,238,996,247 bytes of source hashing per process
(admission and each worker) and the same again for each weight load (2x in load
and 2x in restart); 450,061,586 bytes of environment-closure hashing in
admission, load and restart; plus the checkpoint and teacher cache at restart.
Physical page-cache reuse is not measured and is not subtracted.

Write accounting: a successful wait-based read persists three pending
descriptors (initial pending, dispatch reservation, terminal), a waiting read
that times out persists two, and a read whose own dispatch fails persists
three, so three is the exact unconditional bound and every stage reserves 3 per
read. Each explicit bounded caller resume persists at most two (one when it
attaches to an in-flight ticket, none when refused before dispatch), so 2 of
resume headroom is reserved per read; the read's own internal resume is inside
that 3 and is not charged again. There is no automatic retry. The total is
computed from all 18 stage caps by the protocol module and re-checked by the
validator against `total_writes_value`, so it cannot drift from the envelope.
The count also covers one observed source-cost receipt per closed stage and the
receipts written outside a stage.

Artifact arithmetic: 2.90M reader parameters across two sites plus a 1.32M
parameter compressor, all float32, plus two AdamW moment buffers, bounded at
96 MiB for the checkpoint. This is a cap, not a claim of fit.

Per-world probe arithmetic: 32 worlds x (2 parity + 6 control + 48 generated)
backbone forwards, 3 source forwards, 4 canonical-source tokenizer passes (the
release rebuild clears C before the durable read re-checks provenance), 4
release writes (origin pointer, payload, manifest, publication pointer), 3
descriptor writes for the durable read plus 2 of resume headroom, i.e. 9
declared writes per world, plus the two stage receipts.

Resource envelope, unchanged: one exclusive worker, 50 GiB sampled MPS driver,
512 MiB sustained swap growth, 15% minimum host free, 2 s host samples, 0.25 s
driver samples, 30 s stale-telemetry stop. Driver and RSS overlap and are
reported separately from host-free and swap.

## Differences from v11 (complete)

1. Value producer: frozen canonical-source decoder hidden states followed by a
   live trainable compressor, instead of a trainable encoder over
   template-framed text.
2. Source input: canonical span text only, no chat template, no special tokens,
   explicit zero-based positions, extensions disabled by argument and audited.
3. Bounded in-process feature cache C (32 MiB / 64 entries) that is never
   serialized; cleared on correction, revocation, source-free transitions,
   release publication and checkpoint restore.
4. Four ordered slots per span with the learned order added exactly once by
   `ContextualMemoryV12.snapshot`; intervention controls bypass `snapshot` and
   add order exactly once themselves.
5. New append-only work ledger with the v12 vocabulary and dual-clock deadlines.
6. Re-derived stage envelope (6835 s, 500/900 per fit prefix) instead of 7080 s.
7. Complete v12 checkpoint binding the feature, adapter and tokenizer contracts;
   no v11 checkpoint migration.
8. Teacher cache bound to the feature contract identity as well as the objective,
   base and fitting-data digests.
9. `slot_policy_v12` declared beside the inherited
   `fixed_eight_masked_padding_v5` field that the sealed validators require.
10. v12 producer-release corpus namespace. The release is now the **complete
    representation release** (compressor + order + reader + feature contract),
    so any contextual-path update invalidates previously durable values and a
    fresh release must be republished before durable reads are used again. No
    projection or value cache is presumed valid across an update.
11. Restart recomputes source features from a cleared cache and compares the
    recomputed feature contract and adapter content binding with the checkpoint
    binding. The v12 checkpoint records the full release, its explicit alias and
    the compressor-only release, and refuses any mismatch before mutating
    weights.
12. New data manifest with an explicit exposure contract.
13. The adapter contract is content-bound: the exact base content digest is
    supplied as the already-verified full original parameter digest, so a
    different backbone with identical geometry is refused without hashing the
    original weights a second time.

Unchanged: official BF16 source and revision; reader sites 3/7 and placement;
eight masked ordered slots with a two-span budget; teacher top-64 KL at
temperature 1 with zero CE mix; NativeFusedAdamW at lr 1e-4 and seed 17; 512
fitting updates plus one restart update in a fixed round robin; four fixed probe
prefixes with no best-prefix selection; per-group eight-world fitting and
utility gates; calibration 8 / fitting 32 / development 16 on the same rows;
generation template, scorer, context and token limits; zero persistent-P
updates and no promotion; physical resource limits and the 7200 s ceiling.

## Commands

Focused tests (CPU only, no weights, no downloads):

```
.venv/bin/python -m pytest -q tests/test_intrinsic_budget_v12.py \
    tests/test_intrinsic_coverage_v12.py tests/test_intrinsic_native_v12.py
```

Environment only (qualified interpreter; no weights, no model import):

```
.venv-spark-bf16/bin/python scripts/run_spark_intrinsic_coverage_v12.py \
    --packet reports/intrinsic_memory_native_v12/native_packet_two_v12.json \
    --environment-only
```

Review (environment, code/data closure, predecessor, reused data, bindings;
still no weights):

```
.venv-spark-bf16/bin/python scripts/run_spark_intrinsic_coverage_v12.py \
    --packet reports/intrinsic_memory_native_v12/native_packet_two_v12.json \
    --review
```

Future launch — **guarded and unexecuted.** It requires an external receipt at
`reports/intrinsic_memory_native_v12/authorization_two_v12.json` that this
preparation deliberately did not create, and a fresh output root:

```
.venv-spark-bf16/bin/python scripts/run_spark_intrinsic_coverage_v12.py \
    --packet reports/intrinsic_memory_native_v12/native_packet_two_v12.json \
    --authorization reports/intrinsic_memory_native_v12/authorization_two_v12.json \
    --launch
```

## Review checklist before any authorization

1. Integrated CPU producer/reader workflow, its gradients and its fresh-process
   continuation must exist and pass for this mechanism. The earlier CPU evidence
   predates the v12 files and does not validate them; the repaired producers
   committed at `6ba07e1b` are bound by this closure, and my three focused files
   pass at 64 tests against these bytes.
2. Independent review of these exact bytes, then the packet hash is sealed.
3. `--environment-only` and `--review` must pass under the qualified
   interpreter; neither has been run against the current packet.
4. Only then may a truthful receipt be bound to the standing instruction and
   this exact packet. No old receipt bytes cover changed code.
5. Exactly one launch. A failure closes its dependent stages; it does not
   license editing this packet and relaunching it.

Remaining gaps are listed honestly: useful recall, native density, persistent
learning and RSI are separate conclusions; native uninterrupted-update
equivalence is not measured; the source-representation hypothesis is not proven
by a fitting failure or success.

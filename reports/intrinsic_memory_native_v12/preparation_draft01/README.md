# Intrinsic memory v12 native preparation

Status: **prepared, not launched, not reviewed, not sealed.** No original weight
was loaded, no development label was parsed, no authorization receipt exists and
no native attempt was consumed. This directory holds preparation and review
material for the separately versioned v12 source-feature successor to the
consumed v11 two-site attempt.

The executable closure in `implementation_v12.json` binds the producer modules
as they exist at preparation time. Those five modules were still under repair
when this packet was prepared, so **the closure must be regenerated after the
repaired producer bytes are reviewed and sealed.** A stale closure fails closed:
`--review` recomputes the static import closure and refuses on any difference.
To re-prepare, remove only the preparation outputs listed below (never an
attempt directory) and run the preparation script again:

```
rm -f configs/spark_intrinsic_coverage_v12.json \
      reports/intrinsic_memory_native_v12/implementation_v12.json \
      reports/intrinsic_memory_native_v12/native_packet_two_v12.json \
      reports/intrinsic_memory_native_v12/preparation.json
rm -rf data/intrinsic_memory_coverage_v12_20261029
.venv/bin/python scripts/prepare_spark_intrinsic_coverage_v12.py
```

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
| prepare | 60 | 32 | 32 / 32 | 0 | 0 | 32 | 225 |
| teacher | 90 | 288 | 0 / 0 | 0 | 256 | 0 | 34 |
| fit_128..512 (each) | 500 | 256 | 128 / 128 | 128 | 0 | 0 | 129 |
| fit_probes_128..512 (each) | 900 | 1888 | 128 / 96 | 0 | 1536 | 32 | 323 |
| checkpoint | 60 | 2 | 1 / 1 | 0 | 0 | 0 | 7 |
| development | 540 | 784 | 16 / 16 | 0 | 768 | 16 | 114 |
| correction | 60 | 25 | 2 / 1 | 0 | 24 | 2 | 16 |
| restart | 140 | 4 | 2 / 2 | 1 | 0 | 1 | 8 |

Totals: stage wall **6835 s** inside the unchanged **7200 s** outer envelope
(365 s outer headroom, never a stage allowance); 9778 forwards; 949 source
feature forwards; 1078 canonical-source tokenizer passes; 3966 prompt tokenizer
passes; 513 backwards; 513 updates (512 fitting + 1 fresh-process
continuation); 7256 generated tokens; 179 retrievals; 2228 declared writes;
194 MiB artifacts (96 MiB envelope + 2 MiB teacher cache + 96 MiB checkpoint);
102,760,448 read bytes at restart; 2 model loads.

Write accounting: a successful wait-based read persists three pending
descriptors (initial pending, dispatch reservation, terminal), a waiting read
that times out persists two, and a read whose own dispatch fails is bounded by
four, so every stage reserves 4 per read. Each explicit bounded caller resume
persists at most two (one when it attaches to an in-flight ticket, none when
refused before dispatch), so 2 of resume headroom is reserved per read. There
is no automatic retry. The count also covers one observed source-cost receipt
per closed stage and the receipts written outside a stage.

Artifact arithmetic: 2.90M reader parameters across two sites plus a 1.32M
parameter compressor, all float32, plus two AdamW moment buffers, bounded at
96 MiB for the checkpoint. This is a cap, not a claim of fit.

Per-world probe arithmetic: 32 worlds x (2 parity + 6 control + 48 generated)
backbone forwards, 3 source forwards, 4 canonical-source tokenizer passes (the
release rebuild clears C before the durable read re-checks provenance), 4
release writes (origin pointer, payload, manifest, publication pointer), 4
descriptor writes for the durable read plus 2 of resume headroom, i.e. 10
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
   predates the v12 files and does not validate them, and the producer modules
   were still in repair when this packet was prepared.
2. Independent review of these exact bytes, then the packet hash is sealed.
3. `--environment-only` and `--review` must pass under the qualified
   interpreter.
4. Only then may a truthful receipt be bound to the standing instruction and
   this exact packet. No old receipt bytes cover changed code.
5. Exactly one launch. A failure closes its dependent stages; it does not
   license editing this packet and relaunching it.

Remaining gaps are listed honestly: useful recall, native density, persistent
learning and RSI are separate conclusions; native uninterrupted-update
equivalence is not measured; the source-representation hypothesis is not proven
by a fitting failure or success.

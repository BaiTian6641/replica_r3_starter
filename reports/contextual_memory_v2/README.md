# Contextual memory v2 — CPU implementation and native readout gate

29 October 2026. Starting HEAD and latest independent review: `a4d11366`.
There were no newer commits at inspection. Implementation: `b2489ba133082e494ec1f94cac7d4d54f7b30ad9`.
The original Spark-X2.6-4B-FP8 lineage and permanent typed spine remain authoritative.
Official FP8, the existing FP8-derived BF16 artifact, and official BF16 are distinct.
No native model execution, download, duplicate checkpoint or full-base export occurred.
Historical conversion manifests, qualification receipts and failed trajectories remain unchanged.

## Implemented behavior

Execution provenance now binds `configuration_spark.py`, `modeling_spark.py`, their
relative dependencies and the exact inventory of local executable files in
[`spark_derivative_execution_manifest_v1.json`](../../configs/spark_derivative_execution_manifest_v1.json).
Missing, changed, added or symlink-substituted executable code is refused before
tokenizer/model loading, including metadata-only review. The protocol and external
receipt must bind the execution manifest. This inventory supplements conversion
history. Portable tests construct synthetic source trees, shards and source/Git
fixtures; they neither require ignored checkpoint files nor weaken production checks.
Full source validation is an explicit provisioned integration gate.

Hot P authority is separate from the most recent cold publication. Version-2
journal events bind the exact latest authority digest and sequence; that authority
binds hot clock, lineage, encoder and weight/moment payloads coherently. Cold
publication references are checked against their own manifest. Multiple dirty
writes recover after SQLite deletion without comparing hot lineage to stale cold
lineage. Missing/corrupt newest authority, interrupted payload/journal writes and
revoked dependencies remain quarantined. Healthy unrelated owners and identity
loss protections remain intact; a degraded diagnostic is not available P state.

[`spark_competence_protocol_v1.json`](../../configs/spark_competence_protocol_v1.json)
introduces a separate, pre-fit calibration split with eight independent synthetic
worlds. Pass requires 8/8 exact normalized generated answers and zero false recall.
Original weights are frozen, extensions disabled, thinking disabled, greedy EOS
decode, context at most 2,048 and generation at most 32 tokens. Data, scorer,
template, tokenizer/source identity and effective budgets are bound before results.
Incomplete evidence or truncation refuses learning. Development opening validates
the full receipt; calibration does not open development or sealed-final labels.
The older development baseline also checks all twelve full-context rows.

[`contextual_memory.py`](../../src/replica_lab/contextual_memory.py) implements a
separate v2 producer/reader from the v1 design contract. Canonical sources are
bounded at 8,192 UTF-8 bytes and split losslessly into spans of at most 256 bytes.
A one-layer byte transformer emits four ordered FP32 width-256 slots per span;
slot attention is restricted to each quarter after full-span contextualization.
Source order, offsets, digests, scope, availability, validity, corrections,
producer release, representation and codec identities accompany durable values.
The existing eight-slot path selects at most two spans. Stored slots enter the
spine memory reader directly at sites 3/7. Learned order embeddings make shuffled
values an observable control. Sites 23/31 are a separately measured follow-up.
Hash-derived keys/values remain labelled engineering controls, not contextual values.

Original attention, positional behavior, gated attention, GELU MLP and tied output
are preserved. Only the producer, ordered reader and memory ports are trainable,
with exactly one NativeFusedAdamW owner per tensor. The live training path preserves
gradients through the encoder and zero-initialized output projections. Serialization,
inference decisions and frozen query features explicitly detach. Durable reads
validate producer version, scope/time, correction, generation, revision and payload
identity again at injection. Exact canonical-text/span fallback separately charges
bytes, tokens and calls; its prefill and generation count toward total work.
Extension-only weights and optimizer state reload without exporting the base.

`FrozenContextualP` provides one separately owned P1 expert behind useful-readout
and fresh-process gates. It freezes phi and supports learned/reset P under the same
reader, idempotent teaching, correction quarantine and authenticated reset. Its
CPU fixture is numerical plumbing only. Teacher-forced updates are not source-free
evaluation evidence. The native longitudinal eleven-arm P experiment is not yet
qualified or executed; the repaired original eleven controls remain preserved.
No expanded bank, learned local-writer evaluation, nested clocks or RSI is admitted.

## Measured evidence

- Focused provenance, storage, recovery, contextual, calibration and profiler tests:
  **135 passed in 15.82 s**; [`focused.log`](focused.log).
- Independent CPU diagnostic: disabled parity; nonzero producer gradients after
  output projections begin updating; unchanged frozen base; exact fresh-process
  logits and exact optimizer continuation. Three numerical fit updates are not
  an accuracy result. [`cpu_diagnostics.json`](cpu_diagnostics.json).
- Integrated CPU protocol: **0/8 calibration**, exit 2 as declared; zero fit,
  development or optimizer work; learning labels stayed closed. The failure and
  raw calibration answers are preserved in [`cpu_gate_failure.json`](cpu_gate_failure.json)
  and [`cpu_competence.json`](cpu_competence.json). This supersedes any implication
  that the old 0/12 full-context result should permit more learning.
- Full source integration: all eight derivative shards, parent lineage/weight,
  tokenizer, config and executable inventory validated; no model loaded.
  [`native_readout_packet.json`](native_readout_packet.json). Only external derivative
  authorization is absent from launch bindings. Prior numerical qualification is
  provenance evidence, not qualification of this new experiment.
- Clean-checkout full regression and established verification results are recorded
  in `verification_summary.json` when complete; no pending check is counted as passed.

The CPU diagnostic took **3.147938 s**, with parent peak RSS **400,572,416 bytes**,
child peak RSS **406,847,488 bytes**, and **12,265,605 bytes** of artifacts measured
before the final report. It charged 180 producer input bytes / six encoder calls,
24 slots read, six payload reads / 4,638 bytes and four revalidations. Producer
dispatch wall time was 5,684,708 ns and retrieval wall time 1,877,042 ns. These are
CPU fixture measurements, not native latency or learning benefit. Host free was
sampled at 92%; swap querying was unavailable in the sandbox. Native driver use
was not measured. Do not add RSS and shared-driver counts as independent memory.

## Reproduce CPU work

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python scripts/verify_contextual_memory_cpu.py --output runs/contextual_memory_v2/cpu03
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests:scripts .venv/bin/python scripts/run_spark_persistent_learner_v2.py --stage all --backend cpu_contextual_fixture --device cpu --fixture-context-tokens 256 --fixture-max-new-tokens 2 --output runs/contextual_memory_v2/gated_integrated03 --config configs/spark_persistent_learner_protocol_v2.json --prepared data/spark_persistent_learner_v2
```

The second command is expected to refuse dependent learning on the current random
tiny fixture. Independent numerical/storage diagnostics stay available. All raw
runs remain under `runs/contextual_memory_v2/`; no historical runs were deleted.

## Next executable experiment

Review [`native_experiment.json`](native_experiment.json), the exact protocol,
source integration packet and [`contextual_readout_protocol_v2.json`](../../configs/contextual_readout_protocol_v2.json).
The review packet contains the command and complete bindings, not authorization.
There is no existing receipt to reuse. An external issuer must supply the missing
receipt for the derivative identity, new execution manifest, code/protocol hashes,
`execution_stage=readout` and fresh `runs/contextual_memory_v2/native_readout01` output.
Do not forge, broaden or treat this document as that receipt.

One exclusive MPS worker performs calibration first. Failure preserves diagnostics
and stops. On success, 32 fixed contextual readout updates precede opening eight
independent development worlds across contextual, selected exact, wrong, shuffled
and absent controls (40 responses). Pass requires at least 6/8 contextual correct,
zero false recall, at most one case below selected exact and strictly better than
all three negative controls. A fresh process reloads extension/optimizer, checks
exact logits and performs one actual continuation update. Frozen-base integrity
and finite reader/producer gradients are mandatory. No P update occurs in this stage.

The added stage has at most 1,571 forwards, 33 backwards/updates, 48 generation
requests, 1,536 generated tokens, two model/tokenizer loads and 3,241,216 processed
positions. Conservative whole-protocol reserves additionally retain the historical
11-arm schedule: 10,236 forwards / 21,121,056 positions / 41 updates / 195 checks.
These upper bounds are not measurements. The outer guard retains 7,200 seconds,
256 MiB live artifact cap, no retries, 50 GiB sampled driver, 512 MiB sustained
swap growth and 15% minimum host free. Parent and supervised-child samples share
an append-only telemetry stream; the parent releases its model before the child.

Useful readout must be qualified and frozen before a separate P protocol isolates
learned versus reset P with the same reader/evidence. Keep the 11 controls; remove
E/S/C and answer-bearing I or auxiliary state for source-free generated tests.
Use distinct teaching/future queries, independent worlds and rule families,
chronological corrections/revocations, delayed retention, repeated restarts and
rollback. Score accuracy, false recall, retention, generated tokens, total latency,
teaching/retrieval/replay cost and physical memory. Verified experience must reduce
future reasoning cost without reducing accuracy before claiming reasoning reuse.
These longitudinal measurements remain unrun; do not infer them from parameter changes.

Capacity remains P1 until a measured bottleneck. The local writer's differentiable
inner update stays separate from imperative NativeFusedAdamW and the direct-CE
reference. Require repeated learning/retention/correction/rollback and independent
candidate evaluation before additional clocks, writer expansion or RSI. Preserve
the prior working version. Approved identity is separate from task reward; its
bounded active core and task state take priority within measured budgets. Archival
history uses slower tiers and explicit recall. Directory labels do not establish
physical GPU residency. E/S/P/phi/C/I/K ownership and future specialist-model and
sensor/actor interfaces remain intact; native multimodal work remains deferred.

## Changed files

`changed_files.json` lists this increment's implementation, synthetic data, tests,
protocols, handoffs and current binding metadata. It excludes unrelated local work.

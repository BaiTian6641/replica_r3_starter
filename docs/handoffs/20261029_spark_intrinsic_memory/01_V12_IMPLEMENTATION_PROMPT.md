# Prompt: v12 frozen-source producer and CPU integration

Work in `/Users/flare/replica_r3_starter`. First read
`docs/handoffs/20261029_spark_intrinsic_memory/STATE_AND_BOUNDARIES.md`, the
active root handoff, architecture chapter 28 and the v11 experiment. Inspect HEAD,
newer commits and dirty files. Never read, import, edit or stage
`src/replica_lab/runtime_pilot.py`. Preserve unrelated work and sealed v3–v11
execution files. This assignment is CPU implementation, not native execution.

Own the five v12 WIP modules in `wip_manifest.json`, new v12 unit/integration
tests, synthetic fixtures and a new CPU workflow. Their current status is syntax
only; do not trust copied code or claim earlier 562/29-test receipts cover them.
Coordinate changes to shared dependencies before editing. Prefer a separately
versioned adapter or wrapper over modifying an old bound implementation.

## Implement the declared mechanism

Canonical source text must be bounded, ordered and provenance-bearing. Only an
eligible exact canonical span enters the source encoder: no query, gold answer,
development label, task answer cache or generated explanation. Bind source ID,
record/revision/generation, epoch, scope, cutoff, token sequence, positions,
tokenizer, frozen original digest and executable source contract.

Use the existing frozen original backbone in evaluation mode to extract source
hidden states with memory, spine and other extension branches disabled. Preserve
its pretrained numerical behavior. This source-only forward is an intentional
`no_grad` boundary. The original model must not become a registered child of the
producer or enter the optimizer/checkpoint through aliases.

The trainable compressor consumes detached frozen hidden states through live
torch tensors. Keep normalization/projection/contextual computation and four
ordered width-256 value slots differentiable. Preserve the two-span/eight-slot
selection budget and masks. Store these contextual slots directly and feed them
directly to the memory reader. Keep hash-derived values explicitly labeled as
engineering controls. Do not detach the trainable path via NumPy, hashing, disk
serialization, `.item()` or a cached serialized value.

Start at reader sites 3/7. Queries come from block input; residuals are added
after the complete block. Preserve gradients from early readers through later
frozen blocks; frozen parameters do not justify wrapping the downstream path in
`no_grad`. Keep GELU, gated attention, positional behavior and tied output intact.
Pre-MLP fusion and 23/31 sites are separately measured follow-ups.

Use fresh producer/representation/codec/checkpoint identities. A reader must
refuse stale producer releases, mismatched source contracts, missing/corrupt
authority, wrong scope, future/expired records and mixed generations. Preserve
explicit unavailable/pending/quarantine results. Canonical-text fallback must
use eligible exact spans with a separate token/I/O/latency charge; it must not
silently turn a failed contextual arm into the exact-text arm.

## Close prototype risks

Inspect these concrete areas rather than assuming they are correct:

1. Importability and copied v4/v5 assumptions; clean stale names/comments only in
   new files. Verify all checkpoint manifests and module/state keys.
2. Feature-provider binding to the exact adapter and immutable source/tokenizer/
   feature contract. Refuse substitutions even if tensor geometry matches.
3. Producer-release caching: parameter stamps alone must not hide a changed
   provider contract or tokenizer. Full source hashes remain a native admission
   requirement; mutation stamps are only a supplementary guard.
4. Real pre-dispatch budget admission for source forwards, tokenizer/input work,
   cache misses and recomputation after restart. Counters after the fact are
   insufficient. Give the CPU fixture an explicit accounting provider; do not
   permit an unguarded native fallback when no ledger is supplied.
5. Bounded cache lifetime and ownership. The proposed 32 MiB/64-entry feature
   cache belongs to C, is not a serialized learning artifact and does not grant
   source authority. Charge misses, copies and peak residency. Invalidate or
   clear on correction/revocation, identity changes, close and source removal.
6. Record validation before every consumption and again after delayed cold I/O.
   Correcting a source must invalidate its feature and projected caches even
   when the old payload remains on disk. Do not discard unrelated healthy state.
7. Release publication and crash recovery: bind payload, producer, source pins,
   authority sequence and generation atomically. Rebuilding SQLite cannot
   manufacture authority or revive obsolete values.

Give each intended trainable parameter exactly one NativeFusedAdamW owner.
Verify named identities and object identities; tied/shared modules must not be
registered or stepped twice. Original parameters have zero optimizer ownership.
Declare inference/durable detachment separately from training-time live tensors.

## Required meaningful CPU evidence

Use synthetic sources, a tiny random BF16 backbone with all 36 blocks available,
and explicit synthetic tokenization. No pretrained checkpoint or download.

- Disabled-extension exact parity and actual computation skipping; source-feature
  extraction cannot accidentally read intrinsic memory or the external spine.
- On step one, explain expected zero upstream gradients from zero output
  projections. After outputs begin learning, response loss must reach the
  compressor, ordered slots and every active reader, including early readers.
  Check finite values, missing parameters, original frozen integrity and owner
  counts. Do not demand a nonzero upstream gradient before it is mathematically
  possible.
- Correct, wrong, shuffled, zero and absent slots exercise the intended distinct
  data paths. Engineering influence is not a semantic recall claim on random
  weights.
- One immutable snapshot serves all active readers. Reader count must not
  multiply retrievals, writes or state copies.
- Oversize/empty token input, scope/time violations, stale producer/codec/source
  identities, correction, revocation, cache poisoning and mixed releases fail
  before consumption.
- Hot/warm/cold transitions, cold pending/resume, correction during I/O, corrupt
  payloads, missing authority and interrupted publication preserve fail-closed
  behavior and availability of unrelated healthy state.
- Live training versus durable inference logits agree under the declared
  precision/tolerance. Complete producer, order, reader and optimizer reload in
  a fresh process restores logits and performs an actual update. Compare with
  uninterrupted continuation on the CPU fixture, including optimizer moments.
- Reload refuses a different feature/source contract and never exports original
  weights, cache contents or answer-bearing task state as extension parameters.

Produce an integrated CPU workflow with raw logs, exact code hashes, tests,
resources and artifact inventory. Run clean source-free verification after the
coordinator integrates your changes. Record exactly what is synthetic.

Deliver implemented code, tests, a clear detach/ownership diagram or concise
description, remaining limitations and the new work-accounting interface for
the native operator. Do not open v11 development labels, prepare a fabricated
native success result, or claim native v12 readiness merely because CPU passes.

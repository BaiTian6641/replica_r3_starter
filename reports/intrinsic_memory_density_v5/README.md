# Spark 2/9/36 intrinsic-memory feasibility preparation

**Current successor:** [v11 broader teaching](../../docs/experiments/20261029_Spark_Memory_Coverage_V11.md) is implemented with
fresh data and unchanged per-group gates; the exact official-BF16 control is
prepared. V10 learned fitting but failed useful development at 5/8 contextual
versus 8/8 exact text. CPU 2/9/36 integration passes; native density and persistent
learning remain gated. Standing testing authorization applies to bounded new
packets, with zero retries on consumed outputs. The material below preserves its
historical scope and is not the current launch instruction.

The [v5 protocol](../../configs/intrinsic_memory_density_protocol_v5.json) and
[review packet](native_comparison_packet.json) prepare a separate experiment.
They grant no native authorization. The existing two-site v3 attempt receipts
cannot authorize any density arm. This is a feasibility study of generated
recall, resource use and correction behavior; no density superiority, persistent
learning, capacity expansion or promotion is established.

The layouts are 3/7, the nine full-attention sites, and all 36 sites. The current
placement remains block-input query plus post-block residual. Each arm uses the
same FP8-derived BF16 source identity, qualified environment, eight calibration
worlds, eight fit worlds in four fixed passes, eight development worlds, eight
new chronology worlds, four slots per span, eight selected slots including
masked padding, scorer, templates and generation limits. Producer initialization
and per-site reader initialization are predeclared. The original model stays
frozen; ordinary optimization has one NativeFusedAdamW owner per parameter.
The external spine remains independent.

Each arm must pass 8/8 full-context competence with zero false recall and no
truncated evidence before its 32 fit updates. Frozen fitting precedes development
and chronology labels. Contextual readout must reach at least 6/8, zero false
recall, within one correct answer of exact text, and beat wrong, shuffled and
absent values. The chronology fixture then tests correction and revocation in
eight separate synthetic worlds using contextual and exact-text controls. It
requires stale snapshots, obsolete retrieval and obsolete replay to be refused,
while recording obsolete generated answers separately. These synthetic facts
are controlled test authority, not verified experience from an external world.
They do not establish independent-family generalization.

| Bound per arm | Ceiling |
|---|---:|
| Generated responses | 80 = 8 calibration + 40 development + 32 chronology |
| Generated tokens | 2,560 |
| Forward calls | 2,595 |
| Backward calls / updates | 33 = 32 fit + 1 fresh-process continuation |
| Processed text positions | 5,354,240 |
| Model / tokenizer loads | 2 each |
| Wall time | 7,200 seconds |
| Extension, optimizer, store and log artifacts | 1 GiB |

The position bound charges every growing-prefix forward: 80 times the sum of
2,048 through 2,079, plus 35 nongeneration forwards of up to 2,048 positions.
The two extra nongeneration forwards are parent and child reload logits. The
child performs one real continuation update; no additional uninterrupted
reference update is hidden in the budget. Reload logits require exact equality
on the same device. A failure is retained and stops the arm.

The shared operation limits allow at most 67 source encodings (including the
two reload canaries), 24 record admissions, eight explicit revocations, 32
retrieval requests and 24 exact fallbacks. They allow one layer-snapshot copy
per forward (at most 2,595), zero layer-hook retrievals and zero layer-hook writes. The obsolete
replay check is a read-only eligibility audit; it adds no replay training. Each
exact fallback charges bytes, tokens and prefill time separately. Generated
prefixes reuse pinned query values, but the current wrapper expands and clones
one layer snapshot on each forward; the sequence-dependent copy bytes must be
measured. Reader density cannot multiply any of these allowances.

The three-arm reserve is 7,785 forwards, 99 updates, 7,680 generated tokens,
six model loads, 21,600 serial seconds and 3 GiB of artifacts. These totals are
preparation limits, not an authorization or evidence that a run fits. Every arm
retains one exclusive native worker, a 50 GiB sampled driver ceiling, 512 MiB
sustained swap-growth limit and 15% host-free minimum. Reader parameter estimates
exclude the model, producer, activations and runtime buffers. The artifact cap
is higher than v3 because the 36-reader FP32 weights and two moments alone are
about 596.5 MiB; no base export is permitted.

The next executable command is a portable, model-free preparation check:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/validate_spark_memory_density_v5.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests:scripts .venv/bin/python -m pytest tests/test_intrinsic_memory_density_protocol.py -q
```

This validator checks exact input hashes, per-arm shared exposure, budgets,
scoring, resource limits and output identities. It accepts no authorization and
cannot launch a model. It reads no source weights. Synthetic tests need no
checkpoint or native environment.

A dedicated native density runner is still required. It must bind its complete
executable closure, select the complete producer-plus-reader checkpoint, verify
that all loaded original parameters and buffers are actually on MPS before the
first forward, enforce the listed operation/time/artifact limits, and record
physical resource/cost measurements. The sealed v3 runner must remain unchanged.
Useful v3 recall is a prerequisite; each arm then needs a fresh receipt bound to
the finalized packet, runner, implementation closure, data, numerical/scoring
contract, limits and fresh destinations. No native launch command is represented
as ready while this runner is absent.

Before attributing an advantage solely to density, preregister independent rule
families and seeds, matched-parameter readers (including shared projections),
and matched-total-cost arms with fixed wall/token/memory/retrieval limits. Later
formation and hidden-state persistent learning require separate protocols.
Hot/warm/cold placement does not define a learning clock or consolidation.

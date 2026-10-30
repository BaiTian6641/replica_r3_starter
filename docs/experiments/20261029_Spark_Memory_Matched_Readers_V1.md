# Fixed-parameter reader control, version 1

This separately versioned CPU implementation connects the existing contextual
producer and durable store to 2, 9 or 36 backbone access sites while keeping
exactly two reader parameter sets. It does not modify the sealed v3–v11 runners,
their checkpoints or their native budgets. Useful native recall is still a
prerequisite for a native density experiment.

The fixed mapping is `group = (layer // 4) % 2`. Groups initialize from the same
deterministic streams as the existing independent readers at sites 3 and 7.
The two-site control therefore has identical initialization and updates. At
Spark's width 2560, each arm has exactly 2,895,874 unique reader parameters.
The producer and slot-order parameters are unchanged and also counted once.
Groups are registered once; site dispatch does not create module aliases,
checkpoint tensor copies or extra optimizer owners.

The wrapper uses the real v5 producer, source provenance, temporal validity,
correction/revocation validation, masked eight-slot snapshot and durable store.
The placement remains block-input query with post-block residual addition.
One snapshot serves every active site. Retrieval and writes occur outside
backbone forward. A disabled branch skips producer/reader computation. There is
no projection cache. Every active invocation revalidates authority.

Response loss remains connected through live producer tensors and all later
frozen blocks. Output projections start at zero, so the initial producer
gradient is zero; after the first update it must become nonzero. Tests inspect
each reader invocation's activation gradient, not only the accumulated shared
parameter gradient. Durable admission/read and checkpoint serialization detach;
the training path re-encodes pinned canonical text without serialization.

One `NativeFusedAdamW` owns producer, order and two canonical groups. The complete
checkpoint wraps the existing v5 component format in a distinct outer identity
that binds the exact mapping, layout, optimizer scope and component digest.
Reader-only saves and implicit independent/shared or cross-arm migration are
refused. Frozen loads need an explicit training resume.

The [control protocol](../../configs/intrinsic_memory_matched_reader_v1.json)
defines the native admission requirements and later matched-total-cost study.
The primary comparison holds evidence, slots, writer, update exposure, splits,
template, scorer and generation fixed. It measures actual teaching, retrieval,
publication, latency, tokens and physical memory. Sharing controls parameter
count, but also changes reuse: report that interaction with independent readers.
It cannot establish a pure effect of reader density by itself.

The later total-cost comparison first measures fixed and marginal costs on
independent pilot inputs under its own bounded packet, then freezes whole-pass
training schedules and cost tolerance before scored labels. It restores common
initial weights after the pilot and charges pilot cost. Different exposure is
reported separately from the primary equal-exposure comparison. An arm outside
the declared measured-cost tolerance is unmatched evidence. Neither a parameter
estimate nor an operation count is physical cost qualification.

Run the portable engineering suite with:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_intrinsic_memory_matched_reader_v1.py
```

The random 36-block BF16 CPU fixture establishes mechanics only. Native launch
is disabled in this candidate protocol. It does not add P updates, enlarge P1,
perform formation, qualify identity fidelity or advance RSI. The external spine
continues to connect specialists, sensors and actors independently.

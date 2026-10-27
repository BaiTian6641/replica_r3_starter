# Independent nested-text continuation comparison

**26 October 2026 · exact comparison passed · independent A/A and A/B evidence.**
The protocol was committed at `f743c0e`; canonical state, gradient evidence,
runner instrumentation and the comparator are committed at `7febda4`. This is a
small plastic-bank restore/continuation result, not a capability claim and not
qualification of the stopped 381-tensor production resume.

## Declaration and lineage

`configs/nested_text_continuation_comparison_v1.json` preserves the v1 neural,
checkpoint and event bindings. It declares two fresh uninterrupted 28-event
arms and one fresh resume from the preserved 27-event `teach02` parent. Exact
comparison is mandatory: no numeric envelope, no second in-process reference
verdict, and no retry after a mismatch or guard stop.

| Binding | Value |
|---|---|
| Parent checkpoint | `runs/nested_text_nl1_20261026/teach02/full_checkpoint.pt` |
| Parent SHA-256 / bytes | `0042947840f8c93392a421b11a8d63da7e879dbff7770350287a9309346e8f16` / 107,529,763 |
| TextProtocol SHA-256 | `b0249d3a8258baaebba80166cec0c89abb098a2b64ff599ee5dad4c09125c958` |
| Episode manifest SHA-256 | `88da5a54ab4269f2e26be701c8d02651dfac3de5ddb80208a9a84db0cbc48942` |
| Event boundary | 27 committed events, then `correction-0003` at event 28 |
| Legacy boundary digest | `8ba6b866b41beb0b82563c50c2249c9085f46871cc114125efd0d50b615deb47` |
| Legacy final digest | `ed242b006c261aa26b1ca9fc1351b774bcdcfd8f73e0afa62cc6e57e7e9e91a8` |
| Canonical boundary digest | `31d6821fa845b64b1f3af4ffe0d09627eb256bf20902813c64a26d42d5098574` |
| Canonical final digest | `59cd0c520f869fbadcbca0061a7d86bfa76235063618ea3f41e1e6e50cbf4bf9` |

The canonical digest covers every expert parameter and optimizer moment, full
dedup outcomes, coordinator observations/chunks/fingerprints/`latest_sealed_at`,
clock/revision, event fingerprints, route and personality. Long numeric vectors
are packed into deterministic leaves; the parent has 48 parameter tensors,
12,582,912 plastic parameters, 27 optimizer-state entries, 27 observations,
27 chunks and 1,474 packed leaves. The legacy digest remains a separate
compatibility receipt, not the sole exactness key.

## Native processes and resource evidence

All three processes used the sealed source, one exclusive MPS worker, CPU
fallback disabled, offline execution, and the unchanged 50 GiB / 512 MiB / 15%
guard. No heldout or development rows were opened. Only the declared teaching
and correction rows were used.

| Arm | Updates | Guard wall time | Peak sampled driver | Swap growth | Minimum free | Artifact bytes |
|---|---:|---:|---:|---:|---:|---:|
| `continuous_a` | 28 | 13.5428 s | 10,757,210,112 B | 0 B | 78% | 162,233,255 |
| `continuous_b` | 28 | 13.5396 s | 11,815,763,968 B | 0 B | 76% | 162,232,938 |
| `resume` | 1 (`correction-0003`) | 7.6744 s | 10,757,210,112 B | 0 B | 78% | 589,691 |

Each continuous arm wrote a 107,539,619-byte full checkpoint and a
50,347,007-byte P-only state. The resume used the original parent and wrote no
final model checkpoint. All three `parameter_audit.json` files report 48 plastic
tensors and `theta_stamp_unchanged=true`.

The continuous runs measured 122 supervised response tokens, 28 optimizer
steps and 3.19/3.29 s of update time. Canonical-state instrumentation took
2.87/2.86 s and is reported separately from update time. Guard receipts and
host samples are under `reports/nested_text_continuation_comparison_20261026/`.

## Exact comparison

`reports/nested_text_continuation_comparison_20261026/comparison.json` returns:

```text
status=exact_match
exact_status=passed
variation_status=none
resource_status=within_declared_limits
earliest_divergence=null
```

The comparator verified all 24 teaching plus four correction event IDs,
per-event outcomes, feature/target SHAs, transaction fingerprints and pre-clip
gradient norms, all 16 experts and optimizer moments, full dedup/coordinator
state, event/route/personality bindings, artifact limits and every guard.
`aa_states_compared=57` includes the two 28-event histories and the boundary.

- **A/A:** every canonical event state and transaction record is identical
  across `continuous_a` and `continuous_b`; final P-only SHA-256 is
  `03e34ab932aa7b2e0267f7f4e348022916f1e72bb08444514214daaf2c0c2c34`.
- **Boundary:** both continuous event-27 canonical states equal the preserved
  parent and `resume.before_state`.
- **A/B:** both final canonical states equal `resume.after_state`; final legacy
  digests also match. Checkpoint byte length happens to match, but checkpoint
  bytes were not used as an equality gate. Full checkpoint SHA-256 values may
  differ without changing canonical state.
- **Outcome:** `correction-0003` is committed on expert 8 with loss
  `40.0135498046875`, pre-clip gradient norm `1555.8344526266944`, revision 28,
  one transition at `sealed_at=1043.0`, and no retry. Feature, target and chunk
  SHAs match across all three arms.

A second CPU replay produced `comparison_repeat.json` byte-for-byte identical
to `comparison.json` (SHA-256
`584a79263fb2133a9854502cd34d4a013b5d5aa66ff232a150546c9f9f528082`).

## Commands

The three native commands used `scripts/run_capacity_preflight.py` with the
following arguments after `--`:

```bash
# continuous_a / continuous_b (fresh output and guard path per arm)
--mode teach --protocol configs/nested_text_continuation_comparison_v1.json \
  --episodes data/nested_text_nl1_20261026 \
  --model-path checkpoints/minicpm5_source \
  --model-lock configs/minicpm5_sealed_lock.local.json \
  --output runs/nested_text_continuation_comparison_20261026/<arm> \
  --device mps --world-role primary --max-events 28 --no-replay --no-heldout

# resume
--mode resume --protocol configs/nested_text_continuation_comparison_v1.json \
  --episodes data/nested_text_nl1_20261026 \
  --model-path checkpoints/minicpm5_source \
  --model-lock configs/minicpm5_sealed_lock.local.json \
  --output runs/nested_text_continuation_comparison_20261026/resume \
  --device mps \
  --checkpoint runs/nested_text_nl1_20261026/teach02/full_checkpoint.pt \
  --next-event correction-0003 --max-prompt-tokens 256 \
  --max-response-tokens 64 --no-replay --no-heldout
```

The exact CPU comparator was:

```bash
.venv/bin/python scripts/compare_nested_text_continuation.py \
  --protocol configs/nested_text_continuation_comparison_v1.json \
  --continuous-a runs/nested_text_continuation_comparison_20261026/continuous_a \
  --continuous-b runs/nested_text_continuation_comparison_20261026/continuous_b \
  --resume runs/nested_text_continuation_comparison_20261026/resume \
  --guard-a reports/nested_text_continuation_comparison_20261026/continuous_a.guard.json \
  --guard-b reports/nested_text_continuation_comparison_20261026/continuous_b.guard.json \
  --guard-resume reports/nested_text_continuation_comparison_20261026/resume.guard.json \
  --output reports/nested_text_continuation_comparison_20261026/comparison.json
```

## Verification and boundary

The focused nested-text, controller and comparator suites pass **93 tests**.
They cover canonical full-state coverage, divergence localization, gradient
norm requirements, guard failures, missing boundaries, implementation drift,
P-only mismatch and rejection of in-process verdicts. Native arms then proved
the actual independent-process behavior.

**Implemented:** canonical full-state evidence, per-event gradient and
checkpoint lineage, independent A/A and A/B comparator, resume pending-status
and frozen-backbone audits. **Measured:** exact equality and resource results
above. **Passed:** this 16-expert, 28-event plastic-bank comparison. **Not
qualified:** production 381-tensor MiniCPM5 exact continuation, its A/A
numerical envelope, direct-to-device optimizer restoration, longer schedules,
learning quality, retention or promotion. The stopped full-scope direct-restore
at 579,139,011 bytes of swap growth remains a separate failure and was not
rerun.

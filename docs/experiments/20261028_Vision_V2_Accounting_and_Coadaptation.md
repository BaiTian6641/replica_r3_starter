# Vision v2 accounting and conditional co-adaptation preparation

**28 October 2026. Status: versioned CPU preparation complete; native execution
remains unauthorized and unrun.** Implementation commit:
`e8579c049680c0a36636659f7788609eb1eec48f`.
No model was loaded, no optimizer ran, and no development, held-out, V0,
promotion or retry access opened.

## 1. Version separation and v1 preservation

The first v2 draft modified `scripts/vision_smaller_step_v1.py`, invalidating the
sealed v1 packet and readiness implementation pins. The pre-split run recorded
16 failures and six errors. The implementation now lives in
`scripts/vision_smaller_step_v2.py`; the v1 runner is restored byte-for-byte and
v1 config, packet, readiness and evidence are unchanged. No failing v1 test was
re-pinned. The failure/resolution summary is
`reports/verification_20261028/vision_v2_pre_split_failure.json`.

The v2 runner defaults to `vision_smaller_step_lr3e5_v2`. Its packet builder,
readiness checker, launch/review commands and co-adaptation readout normalization
all use that v2 file. A bound-review test covers the successful command path;
missing-receipt tests cover refusal before native imports.

## 2. Derived smaller-step accounting

The v2 amendment preserves the lr 3e-05 hypothesis, prepared04 batch, six
projector tensors, response-only labels, NativeFusedAdamW mechanics and all
optimization/functional gates. It changes accounting, not the training arm.

| Counter | Superseded v1 cap | Derived v2 schedule |
|---|---:|---:|
| Decoder forwards, one wider readout | 2,400 | 2,390 |
| Decoder forwards, campaign | 4,824 | 4,804 |
| Visual observations, one wider readout / campaign | 78 / 168 | 72 / 156 |
| Visual/projector passes, one wider readout | 2,400 | 2,260 |
| Visual/projector passes, campaign | 4,824 | 4,544 |
| Processed text positions, campaign | not consistently identified | 1,380,240 |
| Processed visual positions, campaign | not consistently identified | 890,624 |

Each wider cell has three prepare/sense and two explicit-logits passes. Generated
tokens equal prefill plus decode. The 26 no-image cells have no visual patches;
projector passes therefore equal visual-bearing passes, not every decoder pass.
An observation is exactly one cache hit or miss, and encoder calls equal misses.
Expected runtime counts are zero hits plus 156 misses; the separate cap fields
remain 156 each and are not substituted for expected values.

Processed-text validation now enforces 288 positions per wider pass and 150 per
update/update-batch pass. One-unit drift tests cover decoder, observation,
projector and processed-text counters. Cap declarations and expected runtime
counts remain distinct.

Evidence:

- `reports/vision_smaller_step_launch_packet_v2_20261027/launch_packet.json`
- `reports/vision_smaller_step_launch_readiness_v2_20261027/readiness.json`
- `configs/vision_smaller_step_lr3e5_v2.json`
- `configs/vision_smaller_step_launch_readiness_v2.json`

The readiness result is `readiness_current_launch_refused`: the packet is not an
authorization, the external receipt is absent, the protocol commit is unbound,
and native execution remains closed.

## 3. Conditional co-adaptation control

`vision_coadaptation_control_v2` changes only gradient scope relative to the
smaller-step arm: six projector tensors plus the 16 tensors in SigLIP2 encoder
block 11 and two `vision_model.post_layernorm` tensors. The exact scope is 24
tensors and 18,631,424 parameters. The MiniCPM5 decoder, embeddings, encoder
layers 0-10 and head remain frozen.

The mutable feature service evaluates the frozen prefix under `no_grad` and the
adapted block/post-norm with autograd. Cache entries carry a producer version;
a tensor-version change invalidates them. Training observations always execute
the adapted block and cannot reuse detached features. CPU tests verify exact
scope ownership and producer-change invalidation, not native gradient quality.

The real review and run entrypoints currently refuse with:

- `smaller_step_report_missing`
- `frozen_feature_diagnosis_missing`
- `authorization_receipt_missing`

Raw review: `reports/vision_coadaptation_control_v2_20261027/review.json`.
The run refusal occurs before output creation or native imports. Co-adaptation is
therefore implemented but ineligible; a failed smaller-step report, a preserved
frozen-feature diagnosis excluding competing causes, a full commit and a
separate external receipt are all still required.

## 4. Verification

The workspace suite passed **184/184**: 161 preserved v1 tests plus v2
accounting/launch, co-adaptation and forward-accounting tests.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_vision_smaller_step_v1.py \
  tests/test_vision_smaller_step_launch_readiness_v1.py \
  tests/test_vision_smaller_step_launch_packet_v1.py \
  tests/test_vision_coadaptation_control_v1.py \
  tests/test_vision_smaller_step_v2.py \
  tests/test_vision_coadaptation_control_v2.py \
  tests/test_vision_forward_accounting.py
```

A detached clean checkout at `e8579c04` passed **296/296** combined vision and
DC-P1 tests with zero skips. Receipt:
`reports/verification_20261028/vision_dc_clean_checkout_summary.json`.

Additional observed checks:

```bash
.venv/bin/python scripts/vision_smaller_step_v2.py validate --json
.venv/bin/python scripts/build_vision_smaller_step_launch_packet_v2.py --check
.venv/bin/python scripts/check_vision_smaller_step_launch_readiness_v2.py verify --json
.venv/bin/python scripts/vision_coadaptation_control_runner_v2.py validate --json
.venv/bin/python scripts/vision_coadaptation_control_runner_v2.py review-launch \
  --config configs/vision_coadaptation_control_v2.json
```

The co-adaptation review returned exit 2 as required. The smaller-step
`review-launch` also returned exit 2 because its external receipt is absent.
No native guard was stopped and no MPS work was attempted.

## 5. Remaining gates

Do not infer that the accounting correction improves learning, or that the
conditional runner establishes useful co-adaptation. Numeric losses, native
memory, gradient nonzero/drift results, generated answers, caption quality and
wall/artifact costs remain unmeasured. The next native action, if separately
authorized, is the bounded smaller-step protocol—not co-adaptation. Co-adaptation
can follow only its documented failure and diagnosis gates with a new receipt.

# DC-R1 v2 readout and compressed-channel preparation

**28 October 2026. Status: versioned preparation and CPU tests pass; no native
runner, authorization or native result exists.** Protocol commit:
`ed923e01`. The v1 config, fixture and evidence remain preserved.

## 1. Readout and abstention semantics

DC-R1 evaluates layers 13, 27 and 41 over exact context, selected literal,
compressed record, wrong value and no memory. An accepted layer means at least
one layer produced the exact target; it is diagnostic only. Complete-group recall
requires every declared layer to produce the target. Group abstention requires
every declared layer's actual re-decoded output to be `UNKNOWN`; any other output
makes abstention false. Non-abstention and hallucination are recomputed from
actual generated/re-decoded text, never host flags.

Case IDs, answer-bearing routing metadata and compressed codes cannot be used as
answer indices. Empty generation earns no credit, wrong-value and corrupted
controls must fail, and sealed-final labels require a valid admitted development
report first.

## 2. Compressed channel boundary

The native compressed input is only `codec_id` plus one integer code. The host
codec table is not visible to the model. `decode_compressed_literal` reconstructs
the canonical bucket literal for 75% of development and sealed-final cases, but
that is a host reconstruction audit/control—not learned native readout evidence.
Native compressed gates remain `blocked_no_native_compressed_channel_receipt`.

This distinction prevents a host lookup from being misreported as a model that
learned to decode compressed records. It does not establish that a future native
channel can learn the mapping.

## 3. Physical-work declaration

The schedule is sequential per case, condition and layer, with no shared or
batched prefill. The old v1 value 240 is preserved as incorrect forward-call
accounting, not reused as a cap.

| Counter | Declared total |
|---|---:|
| Cases / conditions / layers | 16 / 5 / 3 |
| Generation requests / prefill forwards / layer streams | 240 each |
| Decode forwards | 1,680 |
| Forward calls / generated tokens | 1,920 each |
| Prefill tokens | 30,720 |
| Processed positions | 252,480 |
| Optimizer updates / backward calls / checkpoints | 0 |

Processed positions use `T*L + T*(T-1)/2` for the no-cache growing-prefix path.
The live artifact cap is 33,554,432 bytes, with one model and tokenizer load,
zero downloads and the unchanged native guard.

## 4. Verification and remaining gate

The focused suite passes **24/24**, covering deterministic fixture/versioning,
all-layer abstention, mixed-layer hallucination, wrong-value/no-memory controls,
host reconstruction versus native evidence, sequential work counters and
sealed-final access.

Review-only preflight is `review_ready` with no model, tokenizer, labels or native
work:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  scripts/preflight_dc_r1_native_value_v2_20261027.py \
  --review --receipt reports/dc_r1_native_value_v2_20261028/review_attempt01.json
```

The receipt binds config SHA-256
`308aedb7b5c2258e22a8cdf4e8a0cca82e5e7c227544942c2e64046dab30ec25`
and prepared-manifest SHA-256
`aead85097ad8d9078cbcb34b7c7e9c2d848a72f03d1cda0c1a032d19a01ed1d6`.
No authorization or native compressed-channel receipt is present. The declared
launch policy also refuses an otherwise authorized package that lacks a real
native runner. No useful retrieval, compressed readout, abstention calibration or
persistent-memory claim follows.

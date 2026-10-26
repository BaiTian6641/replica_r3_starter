# Spark residency reconciliation

26 October 2026. This note reconciles the Spark X2.6 4B residency figures with
the optimizer that is actually selected now, and it replaces the uncommitted
Spark rows in `reports/sparkx_2_6_product_refinement_20261025/`. That directory
is left untouched as historical work; nothing here edits it.

Artefacts:

- `reports/sparkx_residency_20261026/calculate_residency.py` — standard-library
  calculator; recomputes every total from the pinned tensor header.
- `reports/sparkx_residency_20261026/residency.json` — generated receipt with
  source SHA256 values, the calculator SHA256, header audit, payload bounds,
  training-state scenarios and the guard comparison.

Reproduce with `python3 reports/sparkx_residency_20261026/calculate_residency.py`.
The output is deterministic. The calculator reads three local evidence files
and nothing else; it downloads no weights or assets and starts no native job.

## Inputs

| Input | Value | Source |
|---|---|---|
| Checkpoint | `XHToken/Spark-X2.6-4B` @ `66e3ae7bee54e0a523d692cb8ef841f35409807b` | tensor index header |
| Serialized tensors | 290, all BF16, 5 shards | tensor index header |
| Unique parameters | 4,112,079,360 | recomputed as the sum of `prod(shape)` |
| BF16 payload | 8,224,158,720 bytes (7.659 GiB) | recomputed and equal to `parameters × 2` |
| Tied head | the language head is tied to `model.embedding.weight`; no `lm_head` tensor is serialized, so the head adds no parameters or bytes | tensor index header |
| Header SHA256 | `05fa1b80b4888f319b2c42233b95ce43b5c1783ed335e830667298a1da91e590` | receipt `sources[0]` |
| Selected optimizer | `NativeFusedAdamW`, `torch.optim.AdamW(fused=True)` | `docs/experiments/20261026_Fused_AdamW_Integration_and_Sustained_Performance.md` |
| Guard | 50 GiB = 53,687,091,200 bytes sampled driver stop, with a 512 MiB sustained swap-growth limit and a 15% minimum host-free rule | guard receipt policy |

The header audit asserts duplicate-name absence, `data_offsets` consistency
against shape and dtype, shard payload and tensor-count agreement, and
`payload = parameters × 2`. Any failure aborts the run instead of writing a
receipt.

None of these figures is a native Spark measurement, and no Spark training run
exists at all.

## Weight payload lower bounds

| Storage | Bytes | GiB | Formula |
|---|---:|---:|---|
| FP8 (8-bit) | 4,112,079,360 | 3.830 | `parameters × 1` |
| BF16 | 8,224,158,720 | 7.659 | `parameters × 2` |
| FP32 | 16,448,317,440 | 15.319 | `parameters × 4` |

The FP8 row excludes per-block scales, padding and dequantization buffers. The
BF16 row is exactly the serialized payload of the indexed shards. The FP32 row
is the materialization used by the current fused optimizer path in the MiniCPM
probe; it is not a Spark runtime measurement.

## Training-state scenarios

Every selected scenario is `parameters × bytes/parameter`, an analytical lower
bound over the declared tensors only; the historical Muon row uses its recorded
`9p + 7·aux` expression instead.

| Scenario | Bytes/param | Formula | GiB | vs 50 GiB guard |
|---|---:|---|---:|---|
| FP32 weight, FP32 gradient, two FP32 moments | 16 | `4+4+4+4` | 61.275 | **exceeds** by 11.275 GiB |
| BF16 weight, BF16 gradient, two FP32 moments, no master | 12 | `2+2+4+4` | 45.956 | 4.044 GiB headroom |
| BF16 weight, FP32 gradient, two FP32 moments, no master | 14 | `2+4+4+4` | 53.615 | **exceeds** by 3.615 GiB |
| BF16 weight, FP32 master, FP32 gradient, two FP32 moments | 16 | `2+4+4+4+4` | 61.275 | **exceeds** by 11.275 GiB |
| BF16 weight, BF16 gradient, two int8 moments (historical, unselected) | 6 | `2+2+1+1` | 22.978 | 27.022 GiB headroom |
| FP32 Muon with int8 momentum plus auxiliary FP32 Adam (historical, unselected) | — | `9p + 7·aux` | 36.656 | 13.344 GiB headroom |

Historical rows carry the historical assignment assumption (auxiliary set =
any non-2D tensor plus `model.embedding.weight`, 335,731,200 parameters) and the
historical `7`-byte auxiliary term, whose per-parameter decomposition was never
recorded. Muon/int8 was never measured on Spark and is not a candidate; the rows
are retained only so superseded numbers stay traceable.

All of these rows exclude activations, full logits over the 131,072-row
vocabulary, gradient clipping and reduction buffers, optimizer temporaries
(fused-kernel scratch, foreach lists, step scalars, block scales and
dequantization buffers), allocator and compiler overhead, data loaders, media,
replay, KV cache, and checkpoint/export copies.

## Verdict

**Full-scope Spark is not admitted locally, and this note does not infer that it
fits.** The reasoning is one-directional: the excluded terms can only add bytes,
so any row already at or above the guard is excluded outright, and a row below
the guard is merely unqualified.

- FP32 all-AdamW over the full 4,112,079,360-parameter scope needs
  **65,793,269,760 bytes (61.275 GiB)** for parameters, gradients and two moments
  before a single activation, logit, clipping buffer or allocator arena. That is
  **11.275 GiB above the 50 GiB guard**.
- Adding an FP32 master copy to BF16 storage is byte-identical at 16
  bytes/parameter and is excluded for the same reason. Storing BF16 weights but
  accumulating FP32 gradients — the usual autocast/AMP shape, and the reason the
  12-byte row assumes BF16 gradients only as an optimistic case — costs 14
  bytes/parameter: **57,569,111,040 bytes (53.615 GiB)**, which crosses the guard
  by 3.615 GiB on the parameter and state inventory alone.
- The one BF16 row that lands under the guard is the most optimistic one,
  **49,344,952,320 bytes (45.956 GiB)** at 12 bytes/parameter, leaving **4.044
  GiB** for every excluded term. A lower bound under a ceiling is not evidence
  that a measured peak stays under it.

So full-scope local admission cannot be inferred. What remains unqualified for
BF16 is the whole activation and logits budget at any sequence length, the clip
path, fused-optimizer temporaries, allocator behaviour, checkpoint peaks, and
whether the driver stays sampled below 53,687,091,200 bytes with the 512 MiB
swap-growth and 15% host-free companion rules. Each of those needs a native
measurement under a matched protocol and code pin; none of it follows from a
lower bound, and none of it is claimed here.

This is consistent with the earlier research notes, which already placed full
FP32 Adam at about 61.27 GiB and outside the local envelope
(`docs/research/sparkx_2_6_4b_20261024/README.md`,
`docs/research/sparkx_2_6_product_20261025/Resources_and_Full_Finetuning.md`).
Those documents are unchanged.

## The MiniCPM 8-byte/param cross-check

The sustained fused AdamW probe measured optimizer state of
**20,134,053,364 bytes** over **2,516,756,480** trainable parameters, entirely
`torch.float32` (`runs/fused_adamw_20261025/sustained01/run_report.json`,
`optimizer_audit.state_tensor_bytes`). That is **8.000000606 bytes/parameter**:
two FP32 moments per parameter plus 1,524 bytes of non-parameter scalars. It
confirms the accounting shape used above — the selected AdamW keeps no FP32
master copy, no int8 moments and no extra per-parameter tensors — and therefore
that the 12-, 14- and 16-byte Spark rows are the right structures.

It does not support extrapolating Spark fit. The probe is a 2.52B-parameter
MiniCPM5 run: its sampled peak reached 43,090,886,656 bytes against a 16
bytes/parameter inventory of 40,268,103,680 bytes, so the excluded terms cost
2,822,782,976 bytes (7.0% above the bound) at physical microbatch 1 with
activation checkpointing. That delta belongs to that model, sequence length,
vocabulary and clip path. It must not be scaled to Spark, subtracted from the
guard, or treated as a Spark activation estimate.

## Non-claims and unchanged state

- No Spark weights, tokenizer, dataset or other asset was downloaded; the only
  Spark artefact read is the committed tensor index header.
- No native Spark training, export, adapter, inference or memory-guard job was
  run. No weights were loaded.
- The Spark adapter remains `design_only_not_implemented`; FP8/BF16 source
  preparation stays offline-only.
- Dataset admission, the source registry, optimizer selection and adapter status
  are unchanged. The historical calculators and reports under
  `reports/sparkx_2_6_product_refinement_20261025/` are unmodified.
- No figure here is a native fit, allocator peak, throughput prediction,
  convergence result or learning-quality claim.

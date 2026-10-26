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
  backbone-only and combined training-state scenarios, the parsed encoder scope
  declaration and the guard comparisons.

Reproduce with `python3 reports/sparkx_residency_20261026/calculate_residency.py`.
The output is deterministic. The calculator reads six local evidence files and
nothing else; it downloads no weights or assets and starts no native job.

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
| Declared encoder scope | SigLIP2 92,884,224 + SPEAR 93,289,476 = **186,173,700** parameters | parsed from the revision declaration, validated against component declarations |
| Combined revision scope | 4,112,079,360 + 186,173,700 = **4,298,253,060** parameters | computed, not copied |

The header audit asserts duplicate-name absence, `data_offsets` consistency
against shape and dtype, shard payload and tensor-count agreement, and
`payload = parameters × 2`. Any failure aborts the run instead of writing a
receipt.

None of these figures is a native Spark measurement, and no Spark training run
exists at all.

## Encoder scope declaration

The current revision trains more than the language backbone. Line 292 of
`docs/architecture/21_SparkX_2_6_4B_Consolidated_Architecture_and_MiniCPM_Validation.md`
declares that "the current SigLIP2/SPEAR pair alone adds 186,173,700 parameters
and roughly 2.77 GiB under the same FP32 weight/gradient/Adam accounting, before
activations". The calculator parses that sentence instead of restating scout
arithmetic, and refuses to write a receipt unless all of these hold:

| Validation | Result |
|---|---|
| Declared 186,173,700 equals the component sum 92,884,224 + 93,289,476 | passes |
| Declared "roughly 2.77 GiB" matches the all-FP32 16-byte recomputation of 2.774204 GiB within 0.01 GiB; the 18-byte five-copy rate is 3.120980 GiB | passes |
| `vision.encoder_parameters_observed` equals the sealed manifest's `tensor_inventory.vision_parameter_elements` (92,884,224) | passes |
| Architecture doc, product plan and sealed manifest agree on revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` | passes |
| Both component revisions and the manifest seal are present | passes |

Encoder evidence, each hashed into the receipt:

| Component | Parameters | Evidence | SHA256 |
|---|---:|---|---|
| SigLIP2 base patch16-224 vision tower | 92,884,224 | `configs/sparkx_2_6_4b_product_plan_v4.json` `vision.encoder_parameters_observed`; sealed `checkpoints/siglip2_base_patch16_224_source/source_manifest.json` (`tensor_inventory.vision_parameter_elements`, seal digest `ee6274db57b3b590c3b1abf99c849afc349374d9abd9ae693e720b8634065e1d`) | config `2c0cd6ba18031858b14014a3699b479251d5b3bcdf8b327104a02b3f758d467b`, manifest `8b2351f7b29f2bd7d881405dd978ae714a26c0170b7213d133a9e0059a199024` |
| SPEAR base speech/audio v2 | 93,289,476 | `configs/sparkx_2_6_4b_product_plan_v4.json` `audio.candidate.parameters_metadata` (official repository metadata; no sealed tensor index in the repository) | config `2c0cd6ba18031858b14014a3699b479251d5b3bcdf8b327104a02b3f758d467b` |

The declaration is scope metadata for the revision, not a fresh encoder tensor
inventory and not a compiled model. The sealed SigLIP2 manifest describes 208 of
408 tensors as the vision tower, so the paired text tower and the projector are
outside this count; the SPEAR figure is an official metadata count only. The
architecture document is read for its declaration and SHA256 and is never
modified by this work.

## Weight payload lower bounds (backbone only)

These rows cover the 4,112,079,360 indexed parameters only; the combined
revision-scope payloads are in the section below.

| Storage | Bytes | GiB | Formula |
|---|---:|---:|---|
| FP8 (8-bit) | 4,112,079,360 | 3.830 | `parameters × 1` |
| BF16 | 8,224,158,720 | 7.659 | `parameters × 2` |
| FP32 | 16,448,317,440 | 15.319 | `parameters × 4` |

The FP8 row excludes per-block scales, padding and dequantization buffers. The
BF16 row is exactly the serialized payload of the indexed shards. The FP32 row
is the materialization used by the current fused optimizer path in the MiniCPM
probe; it is not a Spark runtime measurement.

## Training-state scenarios (backbone only)

Every selected scenario is `parameters × bytes/parameter`, an analytical lower
bound over the declared tensors only; the historical Muon row uses its recorded
`9p + 7·aux` expression instead. These rows are **not** the full revision
training inventory, which also trains the encoder pair. They remain the
published backbone floors for comparison, and no absolute claim should use them
once the encoders are in scope.

| Scenario | Bytes/param | Formula | GiB | vs 50 GiB guard |
|---|---:|---|---:|---|
| FP32 weight, FP32 gradient, two FP32 moments | 16 | `4+4+4+4` | 61.275 | **exceeds** by 11.275 GiB |
| BF16 weight, BF16 gradient, two FP32 moments, no master | 12 | `2+2+4+4` | 45.956 | 4.044 GiB headroom |
| BF16 weight, FP32 gradient, two FP32 moments, no master | 14 | `2+4+4+4` | 53.615 | **exceeds** by 3.615 GiB |
| BF16 weight, FP32 master, FP32 gradient, two FP32 moments | 18 | `2+4+4+4+4` | 68.934 | **exceeds** by 18.934 GiB |
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

## Combined backbone plus encoder scenarios

The 186,173,700 declared encoder parameters raise the scope to
**4,298,253,060**. Weight payloads at that scope are 4,298,253,060 bytes (4.003
GiB) in FP8, 8,596,506,120 (8.006 GiB) in BF16 and 17,193,012,240 (16.012 GiB)
in FP32.

| Scenario | Bytes/param | Formula | Bytes | GiB | vs 50 GiB guard |
|---|---:|---|---:|---:|---|
| BF16 weight and gradient, two FP32 moments, no master | 12 | `2+2+4+4` | 51,579,036,720 | 48.037 | 1.963 GiB headroom |
| BF16 weight, FP32 gradient, two FP32 moments, no master | 14 | `2+4+4+4` | 60,175,542,840 | 56.043 | **exceeds** by 6.043 GiB |
| BF16 weight, FP32 master, FP32 gradient, two FP32 moments | 18 | `2+4+4+4+4` | 77,368,555,080 | 72.055 | **exceeds** by 22.055 GiB |

The encoder increment over the matching backbone-only row is 2.080653 GiB at 12
bytes/parameter, 2.427429 GiB at 14, 2.774204 GiB at the all-FP32 16-byte rate
and 3.120980 GiB at the 18-byte five-copy rate. The declared "roughly 2.77 GiB"
sentence uses the 16-byte all-FP32 accounting; it is not the five-copy figure.

Beyond the encoder parameters, the combined rows still exclude:

- spatial/temporal bridges and the SigLIP2 projector/resampler: the width
  768 → 1024 and 1024 → 2560 stages;
- lexical/core encoder, vocabulary extensions, native memory machinery and
  spine attachments;
- fast/expert plastic state from the revision 4.2 and 4.3 ledgers, including
  update staging and replay;
- encoder streaming transients, feature queues and per-camera/per-microphone
  session state;
- every activation, logits, clipping, optimizer-temporary, allocator, media, KV
  and checkpoint term already excluded from the backbone-only figures.

**Non-admission.** Two of the three combined rates exceed the 50 GiB guard on
the parameter and state inventory alone. The remaining 12-byte rate leaves
**1.963 GiB** for the entire bridge/projector/fast-state/activation/logits/KV/
media budget, which is smaller than the excluded terms already measured on a
much smaller model. The combined encoder total is itself a declared scope
figure, so even its parameter count carries an assumption that a sealed encoder
inventory would have to replace.

## Verdict

**Full-scope Spark is not admitted locally, and this note does not infer that it
fits.** The reasoning is one-directional: the excluded terms can only add bytes,
so any row already at or above the guard is excluded outright, and a row below
the guard is merely unqualified. The applicable full scope is the combined
4,298,253,060-parameter revision inventory, not the language backbone alone.

- At the combined scope the realistic all-FP32 accounting
  (**68,772,048,960 bytes, 64.049 GiB**) and the autocast-shaped 14-byte
  accounting (**60,175,542,840 bytes, 56.043 GiB**) both exceed the 50 GiB guard
  before a single activation, logit, bridge, projector, fast-state, clipping
  buffer or allocator arena. They are excluded outright.
- The optimistic combined 12-byte rate, **51,579,036,720 bytes (48.037 GiB)**,
  leaves **1.963 GiB** for every excluded term — bridge and projector weights,
  fast/expert state, activations, logits, clipping, allocator, media and KV. A
  lower bound under a ceiling is not evidence that a measured peak stays under
  it, and this headroom is smaller than the excluded overhead measured on a
  2.52B-parameter probe.
- Backbone-only floors remain published for comparison: FP32 all-AdamW over the
  4,112,079,360 indexed parameters needs **65,793,269,760 bytes (61.275 GiB)**,
  **11.275 GiB above the guard**; the BF16 weight/gradient, FP32-moment row
  without a master is **49,344,952,320 bytes (45.956 GiB)**, leaving 4.044 GiB.
  Adding an FP32 master copy to BF16 weights costs 18 bytes/parameter (the
  five-copy plan), **74,017,428,480 bytes (68.934 GiB)**, and is excluded for
  the same reason; the 14-byte FP32-gradient shape is **57,569,111,040 bytes
  (53.615 GiB)**, 3.615 GiB over the guard.

So full-scope local admission cannot be inferred for either scope. What remains
unqualified for BF16 is the encoder-pair parameter count itself (declared
metadata rather than a sealed inventory), the whole activation and logits budget
at any sequence length, the bridges and projector, fast/expert plastic state,
the clip path, fused-optimizer temporaries, allocator behaviour, checkpoint
peaks, and whether the driver stays sampled below 53,687,091,200 bytes with the
512 MiB swap-growth and 15% host-free companion rules. Each of those needs a
native measurement under a matched protocol and code pin; none of it follows
from a lower bound, and none of it is claimed here.

This is consistent with the earlier research notes, which already placed full
FP32 Adam at about 61.27 GiB and outside the local envelope
(`docs/research/sparkx_2_6_4b_20261024/README.md`,
`docs/research/sparkx_2_6_product_20261025/Resources_and_Full_Finetuning.md`)
and which recorded the same two encoder component counts (93,289,476 and
92,884,224) in `docs/research/sparkx_2_6_product_20261025/Resources_and_Full_Finetuning.md`.
Those documents are unchanged, as is the architecture document whose declaration
this receipt parses.

## The MiniCPM 8-byte/param cross-check

The sustained fused AdamW probe measured optimizer state of
**20,134,053,364 bytes** over **2,516,756,480** trainable parameters, entirely
`torch.float32` (`runs/fused_adamw_20261025/sustained01/run_report.json`,
`optimizer_audit.state_tensor_bytes`). That is **8.000000606 bytes/parameter**:
two FP32 moments per parameter plus 1,524 bytes of non-parameter scalars. It
confirms the accounting shape used above — the selected AdamW keeps no FP32
master copy, no int8 moments and no extra per-parameter tensors — and therefore
that the 12-, 14-, 16- and 18-byte Spark rows are the right structures, with 18
reserved for the plan that explicitly adds an FP32 master copy.

It does not support extrapolating Spark fit. The probe is a 2.52B-parameter
MiniCPM5 run: its sampled peak reached 43,090,886,656 bytes against a 16
bytes/parameter inventory of 40,268,103,680 bytes, so the excluded terms cost
2,822,782,976 bytes (7.0% above the bound) at physical microbatch 1 with
activation checkpointing. That delta belongs to that model, sequence length,
vocabulary and clip path. It must not be scaled to Spark, subtracted from the
guard, or treated as a Spark activation estimate.

## Non-claims and unchanged state

- No Spark weights, tokenizer, dataset or other asset was downloaded; the Spark
  artefacts read are the committed tensor index header and the sealed SigLIP2
  source manifest. No encoder weights were downloaded or inspected.
- The 186,173,700 encoder total is the revision's declarative scope statement,
  validated against component declarations and a sealed vision inventory. It is
  not a fresh encoder tensor index, not a compiled multimodal model and not a
  measurement. The SPEAR figure remains official repository metadata.
- No native Spark training, export, adapter, multimodal, inference or
  memory-guard job was run. No weights were loaded.
- The Spark adapter remains `design_only_not_implemented`; FP8/BF16 source
  preparation stays offline-only.
- Dataset admission, the source registry, optimizer selection and adapter status
  are unchanged. The architecture document, the research notes and the
  historical calculators and reports under
  `reports/sparkx_2_6_product_refinement_20261025/` are unmodified.
- No figure here is a native fit, allocator peak, throughput prediction,
  convergence result or learning-quality claim.

---

> **Revision 2 correction (26 October 2026).** The original note above is preserved as historical evidence. The appended correction supersedes its 16-byte five-copy label and its weights-only scope. The current receipt is `reports/sparkx_residency_20261026/residency_v2.json`; `residency.json` is retained byte-for-byte.

## Spark residency reconciliation: 18-byte training state and explicit dtype inventories

**26 October 2026 · Analytical ledger for the SparkX 2.6 4B revision · Not a native fit, allocator measurement, backend qualification or admission decision.**

This note corrects a critical arithmetic error in the earlier residency receipt. When BF16 stored weights, an FP32 master copy, FP32 gradients and two FP32 Adam moments coexist, the training state costs **18 bytes per parameter**, not 16:

$$
2\ (\text{BF16 weights}) + 4\ (\text{FP32 master}) + 4\ (\text{FP32 gradients}) + 4 + 4\ (\text{two FP32 moments}) = 18.
$$

The 16-byte figure is a different plan: all-FP32 weights (the weights themselves are the master), FP32 gradients and two FP32 moments, $4+4+4+4=16$. It is retained as its own row and is never used for the five-copy plan. The earlier receipt's formula string already read `2 + 4 + 4 + 4 + 4` while labelling the rate 16; that contradiction is the defect this revision removes.

The current receipt is [residency_v2.json](../../reports/sparkx_residency_20261026/residency_v2.json), generated by [calculate_residency.py](../../reports/sparkx_residency_20261026/calculate_residency.py). The historical v1 [residency.json](../../reports/sparkx_residency_20261026/residency.json) is retained byte-for-byte; v2 records its SHA-256 under `supersession`.

### 1. Explicit storage inventory

The ledger separates four accounts that earlier reports mixed:

| Account | Contents | Dtype / size | Role |
|---|---|---|---|
| BF16 source artifact | Sealed official checkpoint, 290 tensors, 4,112,079,360 parameters | BF16, 8,224,158,720 B | Source artifact only |
| FP8 source artifact | Sealed mixed-dtype adaptation checkpoint, 470 tensors | F8_E4M3 3,774,873,600 B + BF16 676,697,088 B = 4,451,570,688 B | Source artifact only |
| Fixed-parameter training state | Backbone, declared encoders, projectors, memory modules, phrase encoder | 12 / 14 / 16 / 18 B/param depending on copies | Training storage |
| Dynamic memory state | Bank tensor, metadata, fast `F`, addressing `A`, plastic `P`, workspace, update staging | Declared dtypes | Runtime/plastic state |

The FP8 account is **mixed dtype**, not `parameters × 1 byte`: 180 F8_E4M3 matrices, 180 BF16 scale tensors (2,285,568 B), a BF16 tied embedding (671,088,640 B), BF16 layer norms/final norm (373,760 B) and BF16 attention gates (2,949,120 B). It is kept separate from BF16 source storage and from the optimizer/training account. Nothing in this receipt demonstrates an FP8 loader, dequantizer, autograd path, MPS kernel or backend.

### 2. Component inventory

All fixed-parameter rows are explicit and sum to **4,320,804,612** parameters:

| Component | Parameters | Evidence |
|---|---:|---|
| Language backbone (tied embedding/head) | 4,112,079,360 | Sealed BF16 tensor index |
| SigLIP2 vision encoder | 92,884,224 | Declared in the architecture document |
| SPEAR audio encoder | 93,289,476 | Declared in the architecture document |
| Vision projector $768\to1024\to2560$ | 3,411,456 | Explicit two-stage bias formula |
| Audio projector $512\to1024\to2560$ | 3,149,312 | Explicit two-stage bias formula |
| Memory query/output projections | 3,276,800 | Declared analytical projection count |
| Two-site width-512 reader Q/K/V/O | 10,485,760 | $2\cdot4\cdot2560\cdot512$ |
| Compact phrase encoder | 2,228,224 | $2\cdot12\cdot256^2 + 256\cdot2560$ |

Encoder totals are declared metadata, not a fresh sealed encoder tensor index. Projector, reader and phrase rows are declared analytical formulas, not measured modules. The dynamic account adds a 4,096-record BF16 bank (896 elements per record), 2 MiB metadata cap, FP32 `F`/`A`/`P`, workspace and update staging; it is not multiplied by the fixed-parameter byte rate.

### 3. Scenario arithmetic

| Plan | Copies | B/param | Backbone GiB | Full fixed scope GiB |
|---|---|---:|---:|---:|
| BF16 weights, BF16 gradients, 2 FP32 moments, no master | 2+2+4+4 | 12 | 45.956 | 48.289 |
| BF16 weights, FP32 gradients, 2 FP32 moments, no master | 2+4+4+4 | 14 | 53.615 | 56.337 |
| All FP32 weights/gradients/2 moments | 4+4+4+4 | 16 | 61.275 | 64.385 |
| **BF16 weights + FP32 master + FP32 gradients + 2 FP32 moments** | **2+4+4+4+4** | **18** | **68.934** | **72.433** |

The declared encoder pair is 2.774 GiB at 16 B/param and **3.121 GiB at 18 B/param**. The architecture document's 2.77 GiB sentence uses the all-FP32 16-byte accounting; it is not a five-copy mixed-precision figure.

### 4. Activations, logits, temporaries and phase peaks

These are included so the ledger is not a weights-only lower bound. They are explicit formulas over a declared 256-position, 36-layer, width-2560 workload with activation checkpointing:

| Bucket | Formula | Result |
|---|---|---:|
| Retained layer inputs | $36\cdot256\cdot2560\cdot2$ | 47,185,920 B |
| Attention recompute workspace | $2\cdot256\cdot16\cdot256\cdot4$ | 8,388,608 B |
| MLP recompute workspace | $2\cdot256\cdot10240\cdot4$ | 20,971,520 B |
| Vision/audio feature workspace | declared patch/token counts | 2,052,096 B |
| Logits/softmax/gradient | $3\cdot256\cdot131072\cdot4$ | 402,653,184 B |
| Temporary buffers | clip reduction + optimizer scratch + feature queue | 17,618,762,768 B |

Restore and export are separate phases, not additions to the live training peak:

| Phase | Formula | GiB |
|---|---|---:|
| Restore from BF16 source | BF16 artifact + BF16 weights + FP32 master | 31.804 |
| Restore from FP8 source | mixed FP8 artifact + BF16 weights + FP32 master | 28.290 |
| Export BF16 + FP8 | FP32 master + BF16 export + FP8 staging | 28.168 |

The 18-byte fixed-parameter state plus dynamic state, analytical activations, logits and temporary buffers is **89.463 GiB**, already above the 50 GiB sampled driver guard before allocator fragmentation, compiler workspace and driver overhead. Restore or export staging raises the analytical phase peak further. Nothing here is a measured peak.

### 5. Gates that remain unqualified

- Loader: **unqualified**.
- Dequantization: **unqualified**.
- FP8/BF16 backend and kernels: **unqualified**.
- MPS resource behaviour and allocator peak: **unqualified**; no native run.
- Training admission: **not admitted**.

A lower bound below the guard would still not be a fit, because the excluded terms can only add bytes. Here the central analytical sum is already above the guard, which is a stronger negative statement but still not a measured OOM.

### 6. Reproduce and provision

All residency inputs are repository-tracked: the BF16 index, the FP8 index, the guard receipt and the architecture document. The bundle manifest binds them with roles and SHA-256 digests.

```bash
## verify provisioned input hashes and clean-checkout status
python3 scripts/prepare_sparkx_research_bundle.py --check

## regenerate and verify the receipt (v1 is never written)
python3 reports/sparkx_residency_20261026/calculate_residency.py
python3 reports/sparkx_residency_20261026/calculate_residency.py --check

## deterministic arithmetic and tamper tests
.venv/bin/python -m pytest tests/test_sparkx_residency.py -q
```

If the manifest reports untracked inputs, the checkout is not clean. Development-only verification is `python3 scripts/prepare_sparkx_research_bundle.py --check --allow-untracked`; that override records `clean_checkout_ready: false` and must not be cited as clean replay.

### 7. Non-claims

- No Spark weight payload, model code, tokenizer, dataset, media asset or native runtime was downloaded, loaded or executed.
- Encoder counts are declared metadata; the only sealed parameter inventory is the BF16 backbone index.
- Activation, logit, temporary-buffer and phase-peak values are analytical formulas, not allocator measurements.
- The mixed-dtype FP8 artifact does not establish FP8 training capability or backend support.
- No throughput, convergence, memory fit or training admission is established.

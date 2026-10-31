# Spark fitting-only sequence-margin and gradient-interference diagnostic v17

Status: **attempt05 is predeclared and not launched after independent review repairs.** Attempts01–03 failed before a valid diagnostic; attempt04 was only prepared. Development labels remain closed.

## Fixed scope and parent

- Source: sealed official BF16 Spark and its established tokenizer/original parameter identity. Official FP8 and FP8-derived BF16 remain different identities.
- Parent: V16 content-key attempt01 extension plus optimizer, manifest SHA-256 `a547053a30c58ad2c431b7a3e8a153f543f718a4328cf54f84c74d7acfec085c`, loaded through explicit validated migration. Content-key mode is required.
- Known–known pairs: original fitting pair IDs **0, 1, 4, 5** (`v11_fit_00/01`, `02/03`, `08/09`, `10/11`). IDs are preserved after filtering; no unknown answers or development rows are used.
- Two isolated one-update arms per pair: update left and update right. Each arm restores identical parameters, optimizer moments and CPU/MPS RNG before the next arm and after the intervention sequence.
- Updates use NativeFusedAdamW and mean negative total log-likelihood divided by the actual supervised-token count (answer tokens plus one EOS).

## Measurements

For each world and each arm, preserve:

- Expected-versus-alternative sequence log-likelihood using each candidate's own tokens under the same memory context; answer-token and EOS contributions separately.
- Frozen source features, compressed slots, actual attention queries/keys/values/weights/reads and per-site residuals.
- `wrong_value`, `wrong_read_3`, and `wrong_read_7` interventions. Read overrides use the partner's pre-projection read vector at only the named site; alternative answers use the same intervention, not an additional wrong-value change.
- Per-example gradient norms, dot product and cosine similarity for paired worlds, before any update.
- Margins before and after one update on the world itself or its counterfactual partner.
- Parameter/optimizer/RNG restoration checks, model loads, forwards, backwards, input tokens, source reads, writes and retained tensors.

The analysis must not treat fixed attention positions or weak weight-only color probes as proof of failed addressing. Complete key/value permutation invariance is valid evidence, not a defect to force into failure.

## Exact bindings and envelope

- Protocol: `configs/spark_interference_diagnostic_v17.json`, SHA-256 `b2f671a49d80670b035312d4ad843602384d620dfba10ffc97ea13244edda02e`.
- Packet: `reports/intrinsic_memory_interference_v17/native_packet.json`, SHA-256 `a0e0a55342328f87471ac6ddbf9dfad687d2c915b1b5762c7f56837ce7f94411`.
- Implementation closure: 69 files, SHA-256 `a1fce9a58b75e618ca5cc97283a952cc99076c79d9dd2c490e860d0ca22b85ed`.
- Budget: 128 forwards, 24 backwards, 8 isolated updates, 8 source encodes/source-feature forwards, 8 prompt tokenizations, 40,960 input tokens, 30 writes and 44,040,192 artifact bytes.
- Time: 1,050 reserved stage seconds inside a 1,200-second outer ceiling.
- Resources: one exclusive worker, 50 GiB sampled driver, 512 MiB sustained swap growth, at least 15% host free and CPU fallback disabled.
- Attempts: one, zero retries; separate exact authorization required.

Two CPU tests prove complete pair evidence and exact optimizer restoration. Native packet review and environment-only validation pass without model import or development-label access. No candidate training arm is selected until these measurements are reconciled.

## Attempt01 failure and separate attempt02

Attempt01 stopped before any pair work with `v12 objective/slots/checkpoint identity differs`.
The parent requires `reader_key_mode=content`, but the packet declared it only under
`diagnostic`; `load_native` reads the top-level field and defaulted to `ordered`.
The raw failure is preserved and not retried.

Attempt02 binds `intrinsic_reader_key_mode=content` at both protocol levels:

- Protocol: `configs/spark_interference_diagnostic_v17_attempt02.json`, SHA-256 `4c282b699405b5e5e41ad93d9a8f8e5cb5815c561f2c80dbb5a2608475df4643`.
- Packet: `reports/intrinsic_memory_interference_v17/native_packet_attempt02.json`, SHA-256 `76fdd6afbc77d55fe4534b29e2ffee82a43417fd3c429cce7aff093d75b37423`.
- Implementation closure: SHA-256 `8bf649555f1bf15cc724f3df0af1abbdf704ab36ee22f5240cbab2f0f9ca8cab`.
- Output: `reports/intrinsic_memory_interference_v17/native_bf16_interference_attempt02`; one new attempt, zero retries, separate exact authorization required.

All pairs, measurements, parent checkpoint, budgets, isolated restoration and
development boundaries are unchanged.

## Attempt02 label failure and attempt03

Attempt02 loaded the parent correctly, then failed because pairs 2 and 6 contain
`unknown`. Its completed pair00 artifact and the failure receipt are preserved.
Attempt03 uses the actual known-known IDs 0, 1, 4, 5 (two failing and two passing
references):

- Protocol: `configs/spark_interference_diagnostic_v17_attempt03.json`, SHA-256 `bfd66a6d7adb302b1dc95600fe8d2d6a849dba20d3b2f823ed391fd95ed5ff9d`.
- Packet: `reports/intrinsic_memory_interference_v17/native_packet_attempt03.json`, SHA-256 `f591e8b359e177f113d45cc2f6c10342cb54792e84271ffd8743ce10e5b97f65`.
- Implementation closure: SHA-256 `3647ce51c81595128e93f5ed07be2b19d155c7972a6a583c14d9928b3ffdad64`.
- Output: `reports/intrinsic_memory_interference_v17/native_bf16_interference_attempt03`; one new attempt, zero retries, separate exact authorization required.

The parent, measurements, isolated restoration, budgets and development boundary
are unchanged. No consumed attempt is rerun.

## Independent review and attempt05

The independent reviewer identified stale default pair IDs, nested forward
double-charging, an incorrect gradient dot product and update-token undercounting.
Attempt05 uses the declared pair constant, one forward reservation per model call,
element-wise gradient products and the full encoded token count. It also handles
malformed manifest/record containers without blocking healthy recovery.

- Protocol: `configs/spark_interference_diagnostic_v17_attempt05.json`, SHA-256 `970155e0684f66d5e32439a21e1ef584db3659e4dba2060a0ad5b0e49ed1012d`.
- Packet: `reports/intrinsic_memory_interference_v17/native_packet_attempt05.json`, SHA-256 `db2c5342100e97d94f6ce0360fd16af4e3269c23ca657a8f6dab044308f33d57`.
- Implementation closure: SHA-256 `c19293a718bb793c5730e642e765382201d6710adceae115ccfbff0d8a397818`.
- Output: `reports/intrinsic_memory_interference_v17/native_bf16_interference_attempt05`; one new attempt, zero retries, separate exact authorization required.

Attempt03's closure failure is preserved. Attempt04 was prepared but not launched.
No historical packet or failed receipt is rewritten.

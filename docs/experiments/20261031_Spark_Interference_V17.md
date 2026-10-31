# Spark fitting-only sequence-margin and gradient-interference diagnostic v17

Status: **predeclared and review-qualified; native attempt01 not launched.** This is one bounded diagnostic, not a training comparison or capability claim. Development labels remain closed.

## Fixed scope and parent

- Source: sealed official BF16 Spark and its established tokenizer/original parameter identity. Official FP8 and FP8-derived BF16 remain different identities.
- Parent: V16 content-key attempt01 extension plus optimizer, manifest SHA-256 `a547053a30c58ad2c431b7a3e8a153f543f718a4328cf54f84c74d7acfec085c`, loaded through explicit validated migration. Content-key mode is required.
- Known–known pairs: original fitting pair IDs **0, 2, 4, 6** (`v11_fit_00/01`, `04/05`, `08/09`, `12/13`). IDs are preserved after filtering; no unknown answers or development rows are used.
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

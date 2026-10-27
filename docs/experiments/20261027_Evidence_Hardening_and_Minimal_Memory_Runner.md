# Evidence hardening and minimal memory runner — 27 October 2026

## Scope and authority

Starting HEAD: `81cdf012`. Implementation commit: `f0fff3e4`. MiniCPM5 remains
the baseline and NativeFusedAdamW/`adamw_fused` the default. The 50 GiB driver,
512 MiB sustained swap-growth, 15% host-free, exclusive-worker and
disabled-fallback policies are unchanged. The protected runtime pilot was not
read, imported, modified or staged. **No new native model run, download,
promotion, neural RSI or personality adaptation occurred.**

## 1. Tokenizer-bound fit evidence

`scripts/run_nested_text.py` now rejects the reproduced false pass where
`generated_ids=[EOS]` and `content_ids=[]` appeared with nonempty correct text.
Every fit attempt is decoded before scoring with the sealed tokenizer and the
same decode policy used to create it:

- `replica.nested_text.decode.v1`: `skip_special_tokens=true`,
  `clean_up_tokenization_spaces=false`, stripped outer whitespace, and terminal
  stop tokens excluded before decode;
- sealed repository/revision, lock/config/snapshot and tokenizer-file hashes;
- fast tokenizer loaded offline with `local_files_only=true` and
  `trust_remote_code=false`, after structural/protocol/state checks and without
  `torch.load`, an MPS worker or model weights.

Fit receipt schema is now v2 with a required `tokenizer` block and per-row/gate
`decode_policy_version`. Invalid/out-of-vocabulary IDs, hidden stop tokens,
nonempty text with empty content, decoded-text mismatches, impossible stops and
resealed inconsistent reports fail closed. Positive fixtures use reversible,
consistent token/text evidence. **272 admission/scoring tests passed**, including
real pinned-tokenizer offline decoding. The reviewed current continuation receipt
was migrated to runner `ab41a1ce…` and digest `f379eaa4…`; historical blobs are
unchanged.

## 2. Memory admission and construction/runtime separation

The memory contract now resolves actual checkpoint, fit-report and prediction
artifacts. Admission recomputes per-case correctness, coverage, thresholds and
stale-value rates from authoritative permitted targets and verifies parent,
treatment, optimizer, trainable scope, update counts and resulting state.
Claimed aggregate metrics and placeholder hashes are not evidence.

A trusted all-split construction audit is explicitly identified as
`trusted_construction_audit`. Runtime prechecks parse hashes and allowed training
metadata only. `LabelVault` opens development only after fit admission and
held-out only after development admission, with access spies and no repeated
held-out access. Missing/failed/tampered receipts cannot trigger evaluation-label
parsing or native loading. Reports distinguish dataset auditing, label access and
model evaluation.

Only `learned_value_readout` / `fit_treatment` with its actual 24 updates can bind
a fit receipt. The shuffled and wrong-world control fits each have a separate 24
updates and cannot admit development. Development and held-out receipts repeat
the same per-case/artifact requirements and chain upstream receipt hashes.

## 3. Minimal memory runner and actual CPU result

`src/replica_lab/value_readout_abstention.py` and
`scripts/run_value_readout_abstention_v1.py` implement the declared 22-class
selected-expert readout, fixed exact-address record memory, one event clock per
arm, three independent 24-update arms and all seven controls. They reuse
NativeFusedAdamW, atomic persistence and per-case scoring. Query-time memory is
immutable; corrections, unaffected retention, unknown/wrong-world abstention,
update counts and resulting states are measured separately.

The actual CPU smoke used fresh outputs and the independent `admit` command:

| Check | Result |
|---|---|
| Fit | **27/27 cases**, all declared metrics 1.0, 24 treatment updates |
| Development | **10/10 cases**, all declared metrics 1.0 |
| Held-out | **10/10 cases**, all declared metrics 1.0 |
| Controls | All seven arms in declared order and independently scoped |
| Resume | Byte-identical checkpoint, predictions, fit report, fit receipt and budget report |
| Artifact budget | **29,116,864 bytes**, within amended 32 MiB cap |
| Runner provenance | `scripts/run_value_readout_abstention_v1.py` |

This is deterministic software integration on synthetic features. It does not
establish learned retrieval, open-ended generation, write-through meta-gradients
or general persistent learning. Exact context remains an ungated quality/cost
reference.

The initial output was **29,107,600 bytes**, exceeding the old 16 MiB cap. That
failure is preserved in `reports/verification_20261027/memory_artifact_overrun.json`.
The amendment is explicit: three resume-complete checkpoints total 28,970,835
bytes, plus evidence. The new 33,554,432-byte cap leaves 4,446,832 bytes of
headroom and is enforced after all final report/receipt writes, including the
budget file itself. Admission requires the budget report and actual tree-size
match. **34 memory tests passed.**

Current identities:

- Config: `aa452029185259ca4784b891f8d19b80ce7f4cd27f78de1c8173dbb2d9accfa1`.
- Fixture02 manifest: `7c779d6540be519c5210a08fa1b9dfada52904ca6ae199e4f9a2a249ba581cc0`.
- Treatment fit receipt: `f6defe6348e93b602f61a945fc4662813ee6cfc2b876afa7db018b2bcfb00ee4`.
- Development receipt: `76336b6e95e37c65e7380ff6eef71e429f80d2f5dcf574f8a1428bf5238a1334`.
- Held-out receipt: `e88af3db21333e44848bc196ad096baddb47d5f41e0c1f67dd4729eea5bc981f`.

## 4. Smaller-step vision launch packet

`reports/vision_smaller_step_launch_packet_v1_20261027/launch_packet.json`
contains exact config/implementation/evidence hashes, limits, launch command,
fresh-output rule, protocol-commit requirement and separate authorization
schema. Embedded packet digest:
`9be3a8d2edc7d0b4d819197e22cd316a82affd7b83345cc95c2b2ef7d591a770`.
**85 vision tests passed**, and deterministic `verify` reports `packet_current`.
Preflight refuses before model loading: no authorization exists, the protocol
commit is unbound and no retry is authorized.

The single variable remains lr 3e-5 versus completed 3e-4 with the same parent,
seed 29, batch, six projector tensors and NativeFusedAdamW. Limits remain 4,824
decoder forwards, 168 visual observations, four backward calls, one optimizer
step, 1,200 seconds and 32 MiB including the final report. Optimization and
functional fit gates remain separate. Development, held-out, V0, promotion and
retry stay closed.

Review-only launch packet (do not execute from this handoff):

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONDONTWRITEBYTECODE=1 \
.venv-vision/bin/python scripts/vision_smaller_step_v1.py run \
  --config configs/vision_smaller_step_lr3e5_v1.json \
  --output runs/vision_smaller_step_lr3e5_v1_20261027 \
  --model-path checkpoints/minicpm5_source \
  --model-lock configs/minicpm5_sealed_lock.local.json \
  --authorization reports/vision_smaller_step_lr3e5_v1_20261027/authorization.json
```

The next decision is whether the smaller step keeps update-batch and wider-cohort
NLL within 0.25 nats while satisfying the independent functional fit gate. It is
not a repetition of the completed reproduction and not a claim of conversion.

## 5. Dataset admission and Spark boundaries

The trusted local reviewed caption receipt now validates exactly nine eligible
training children:

- Receipt: `configs/caption_admission_receipt_20261027.json`.
- Digest: `8ca183142d8af706d61ca3c12ba9cf089c4e1e7622228fff33190c612a3cb955`.
- Conditions: research only, no native training/evaluation/retry, and continued
  exclusion of exposed and validation-only rows.

The parent rejection, one mismatch, five unclear, seven validation-only and all
eight exposed rows remain in the 21-row exclusion ledger. The 24 synthetic scenes
remain **diagnostic synthetic evaluation only**, not broad natural-image
validation. All 54 lineage/Spark tests passed.

Spark same-weight reload/base identity now requires `rtol=atol=0.0` with exact
logits and greedy tokens. Historical zero-difference reloads were judged under a
looser threshold and do not prove the stricter rule. A separate candidate
FP8-versus-derivative tolerance (`rtol=0.05`, `atol=0.5`) is explicitly
undeveloped and must be re-derived before any future authorized comparison. The
historical 10.875 cross-arm maximum is not a tolerance or parity result.

The 11,594,609,442-byte extension estimate is small-extension adaptation only.
The complete evolution architecture is not estimated or admitted; full-base
BF16 at 87,256,869,666 bytes already exceeds the usable envelope. Full Spark
training remains blocked. Official BF16, official FP8 storage, dequantized
derivatives and actual native FP8 execution remain distinct. Self-replay remains
a later controlled text-retention ablation, and unqualified visual answers are
never ground truth.

## 6. Verification and preserved rejection

- **272** token/admission tests, **174** source-provenance tests, **34** memory
  runner/admission tests, **85** vision packet tests and **54** data/Spark tests
  passed before the final clean integration suite.
- Continuation historical replay: `exact_match`, zero failures.
- Historical accounting: `replay_matches`, **7,695 / 7,755**.
- Corrected step-1: original `stopped_engineering_failure`, derived
  `step1_reproduction_supported`.
- Response-CE fit remains 0/28, memory v4 remains 21/384 accepted-correct, the
  vision curve remains `curve_no_optimization_progress`, and the caption parent
  remains rejected. The artifact overrun is preserved separately from the later
  budget amendment.

No historical report or pin was rewritten to make a current test pass.

## 7. Remaining blockers and next decisions

- A production fit receipt and useful MiniCPM future-answer memory remain unproven.
  The synthetic CPU readout is not learned retrieval or persistent learning.
- Smaller-step vision remains unexecuted on MPS. It needs a committed protocol
  and separate external authorization before any launch, then optimization and
  functional fit must be decided independently.
- Native FP8 execution, full Spark residency and complete evolution-architecture
  training remain blocked. Small reloads and storage dtype are not training
  qualification.
- Broad natural-image validation, generated visual correctness, causal image
  dependence, neural RSI and personality adaptation remain unrun or unqualified.

The next CPU-side decision is whether this evidence remains reproducible in the
clean committed checkout. Any later native experiment needs a separate protocol
commit and authorization and must preserve these exact limits and admission gates.

## 8. Final verification

The [final verification receipt](../../reports/verification_20261027/summary.json)
records the clean `1702a1ff` suite: **2,804 passed**, zero failures, 95 explicit
artifact/Pillow skips and 21 subtests in 428.443 seconds. Artifact guards pass
over 11,122 files and clean Spark verification passes 436 checks without a
source cache. The earlier missing-manifest and missing-precondition failures are
preserved under `clean_export/tests_attempt01.xml` and `tests_attempt02.xml`.
Tracked metadata fixtures fixed them without skipping or weakening packet tests.

## 9. Copy-ready continuation prompt

> Continue from the current HEAD. Inspect `git status`, intervening commits,
> `AGENT_START.md`, `execution_state.md` and
> `docs/experiments/20261027_Evidence_Hardening_and_Minimal_Memory_Runner.md`.
> Preserve concurrent work and all historical evidence. Do not read, import,
> modify or stage `src/replica_lab/runtime_pilot.py`. Keep MiniCPM5,
> NativeFusedAdamW and the 50 GiB / 512 MiB / 15% native guards unchanged. Do not
> launch native execution from this handoff.
>
> First rerun the 272 token/admission tests, 174 provenance tests, 34 memory
> tests, 85 vision packet tests and 54 data/Spark tests. Replay continuation into
> a fresh output and run `replay-accounting --check` and `replay-step1 --check`.
> Verify the 27/27 fit, 10/10 development and 10/10 held-out artifacts through
> the independent `admit` command, and compare the artifact tree with the 32 MiB
> cap. Keep the original artifact overrun and both clean-fixture failures visible.
>
> Do not claim learned retrieval, open-ended generation or persistent learning
> from the synthetic CPU result. A future native readout needs a new committed
> protocol and independently validated fit receipt before development labels.
> The smaller-step vision `run` command remains review-only: commit its exact
> protocol and issue separate authorization before any launch, then decide
> optimization and functional fit independently. Do not retry automatically.
> Keep the eight exposed rows out of training and untouched evaluation, the 24
> synthetic scenes diagnostic-only, and full Spark/RSI/personality deferred.

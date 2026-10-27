# External caption asset QA — 27 October 2026

## Decision

The existing local 30-image LLaVA-Pretrain tranche moved from metadata research
to actual asset QA, but **training admission is rejected**. No
`training_admitted.jsonl` was written. The seven development rows remain
validation-only. The eight `final_eval_untouched` rows are quarantined after an
accidental semantic-review exposure and are not eligible for training or future
untouched-evaluation claims.

This is an asset-quality decision, not a Spark training admission or a learned
vision result. Spark official BF16, official FP8, dequantized derivatives and
native FP8 execution remain separate provenance/backend cases. MiniCPM5 remains
the baseline.

## Source and immutable identities

- Source: `liuhaotian/LLaVA-Pretrain`.
- Revision: `70f9d1e5e1a697fe35830875cfc7de1dd590d727`.
- QA config: `configs/external_caption_asset_qa_v1.json`.
- Intake config SHA-256:
  `04b5eac5ab22dabcd0ec0a8b73a6321ea28ae840e161f08f07f18903ec22126b`.
- Tranche manifest SHA-256:
  `3e9d9e2428079c12cbe45d006512ce97193280f6f66ec235af78553af7851232`.
- Source card SHA-256:
  `2c0409a377737d7d9aa37c5f72abcf6e6303c6b9b682922527314f7aec5342f4`.
- Local media: 30 JPEGs / 30 root groups / 1,496,105 bytes, each checked against
  its recorded SHA-256 and ZIP member identity.
- Corrected review queue SHA-256:
  `eca352f42068d22c990568ea05faae2bbeeb9d5975d6074fdd9afa28a5902623`.

The QA also verifies annotation/metadata windows, declared full-source hashes,
row/target hashes, provenance and terms fields, duplicate image/caption/root
groups, row IDs, split seals, reserved-family exclusion and label separation.
No network access, download or native model execution occurred.

## Measured QA

| Check | Result |
|---|---:|
| Machine checks | **2,656 / 2,656 passed** |
| Images decoded with Pillow 12.0.0 | **30 / 30** |
| Training candidates | 15 rows / 15 root groups |
| Development validation-only | 7 rows / 7 root groups |
| Sealed final evaluation | 8 rows / 8 root groups |
| Duplicate/split/leakage failures | 0 |
| Focused QA tests | **17 passed, 1 explicit skip** |

Semantic review covered only the 22 non-sealed rows after the QA scope repair.
Among the 15 training candidates, **9 matched**, **1 mismatched**, and **5 were
unclear** because identity, brand, location, event or scent claims were not
visually verifiable. All seven validation-only captions matched their images;
this does not make them training rows.

The formal decision is
`reports/external_caption_asset_qa_20261027/admission_decision.json`.
The corrected machine report is `qa_report.json`; the queue is
`review_queue.json`.

## Sealed-review incident and containment

The first QA template incorrectly required semantic review of all 30 rows. A
contact sheet containing the eight sealed image/caption pairs was generated and
viewed before the defect was found. This was asset review only: no model,
training, scoring or development/held-out evaluation ran. It still invalidates
those rows as untouched independent evaluation data for this workstream.

The exposure and quarantine are recorded in
`reports/remediation_20261027/sealed_caption_review_exposure.json`.
The pre-correction report/queue/template and all contact sheets are preserved;
nothing was deleted. The corrected QA:

- never requires sealed semantic reviews;
- rejects any sealed row appearing in `semantic_reviews`, even with a null verdict;
- redacts sealed prompt/caption/target text in future queues while retaining
  structural hashes and decode evidence;
- excludes sealed and validation-only rows from every training output.

The incident is an independent reason for the formal rejection. A fresh,
independently sealed evaluation tranche is required before any future visual
quality claim.

## Deterministic commands

```bash
# Recompute machine checks and validate the signed rejection.
.venv-vision/bin/python scripts/qa_external_caption_tranche.py \
  --config configs/external_caption_asset_qa_v1.json

# Run the CPU contract tests.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_external_caption_asset_qa.py
```

The script never writes a human decision. It validates the existing decision,
its queue hash, all required non-sealed reviews and the admission scope.

## Next gate

Do not reuse this tranche as untouched evaluation or attempt to admit its
mismatched/unclear training captions. Select a fresh small tranche with immutable
source pins and a separately sealed evaluation root. Perform non-sealed semantic
review before any admission decision, and keep Spark source/backend qualification
independent from this dataset result.

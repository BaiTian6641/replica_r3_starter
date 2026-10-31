# V13 within-fitting-set slot decodability

## Scope

This is **within-fitting-set decodability only**. The compressor trained on these rows. It is not held-out generalization, complete fact preservation, or evidence that Spark's native reader uses the recovered information. Development labels were not read and no parameters were updated.

## Protocol

- Input: mean of the four V13 slot vectors for each known-answer world, 24 rows.
- Training-fold centering, then per-row L2 normalization.
- Ridge regression, regularization 1.
- Unpenalized training-label-prior intercept.
- Whole-pair holdouts, so counterfactual partners remain together out of training.
- One hundred fixed-seed shuffled-label controls with seeds 20261030–20261030+99.
- Exact text and development labels are not used.

## Result

`slot_decodability_run02.json`: baseline **21/24**. Shuffled-label controls: **0–14**, mean 6.24. This reproduces the reported 21/24 baseline, but not the reported 0–9 control range; the different control outcome is retained rather than tuned away.

`slot_decodability.json` is the preserved incorrect first implementation: it normalized before centering and scored 17/24 with controls 1–15. The corrected protocol centers first and normalizes second.

The result supports recoverable information in averaged slots. It does not demonstrate query-conditioned addressing, native behavioral use, sufficient compression, or a mechanism repair. Nonzero slot distances and RMS magnitudes are not used as information measures.

Pair categories and all controls are separately reconciled in `pair_categories.json`: known-known 1/8 for both V12 and V14, known-unknown 7/8 for V12 and 8/8 for V14. Seven V14 known-known failures produce identical answers on both sides. Signed V13 answer-token margins are included; V14 margins are not inferred because full margin vectors were not preserved.

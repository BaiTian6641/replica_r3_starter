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

`slot_decodability_run03.json`: baseline **21/24**. Corrected shuffled-label controls: **0–12**, mean 4.70. Every prediction retains its original `pair_index`, `side` and `world_id`. Preprocessing is fitted inside each fold, and shuffled controls score against the corresponding shuffled held-out targets rather than the original answers.

`slot_decodability_run02.json` retains the earlier center-first implementation with a **wrong control-scoring contract**: it evaluated shuffled predictions against original held-out labels, producing mean 6.24. Its 21/24 baseline is unchanged, but those control scores are not used for mechanism selection.

`slot_decodability.json` is the preserved incorrect first implementation: it normalized before centering and also used the legacy control scoring. The current protocol centers within each training fold, normalizes using that fold's center, and evaluates targets from the same label assignment used to fit the fold.

The result supports recoverable information in averaged slots. It does not demonstrate query-conditioned addressing, native behavioral use, sufficient compression, or a mechanism repair. Nonzero slot distances, fixed attention positions and RMS magnitudes are not used as information measures or proof of failed addressing.

Pair categories and all controls are separately reconciled in `pair_categories.json`: known-known 1/8 for both V12 and V14, known-unknown 7/8 for V12 and 8/8 for V14. Seven V14 known-known failures produce identical answers on both sides. Signed V13 answer-token margins are included; V14 margins are not inferred because full margin vectors were not preserved.

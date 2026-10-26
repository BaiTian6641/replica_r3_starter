# Data sufficiency review evidence

Read the [review](../../docs/experiments/20261026_Training_Plan_Data_Sufficiency_Review.md) for the interpretation and limitations.

- `estimate.json`: prospective distinct-record, repeated-position, inner/outer-update, evaluation-power and conditional compute calculations.
- `input_manifest.json`: 15 committed repository inputs, with bytes, SHA-256 and the observed Git revision. No media or checkpoint copies.
- `external_sources.json`: seven primary research sources, locations and comparison limits, consulted on 26 October 2026. This is a claim ledger, not a downloaded-paper archive.
- `validation.json`: offline evidence identity, independent arithmetic, unit, replay and report-link checks. Passing validates the calculation, not the proposed model's quality.

Run from the repository root:

```sh
.venv/bin/python scripts/estimate_convincing_result_data_budget.py --output reports/local/data_budget_replay.json
```

The calculator requires the pinned local inputs and refuses output replacement. If an input has since changed, use its recorded Git revision in a separate checkout with the review's calculator/config. Do not reset an active checkout to reproduce this report.

Central recipes are 1.2M ordinary records plus 6k episodes for MiniCPM, and 3M plus 15k episodes for Spark. Data quantities are proposed, not acquired/admitted. CUDA rates are inherited assumptions; no training or GPU benchmark was run here. The large factorial audit has an explicit position account and no fabricated GPU-hour estimate.

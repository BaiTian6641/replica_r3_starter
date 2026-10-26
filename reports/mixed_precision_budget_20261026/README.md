# Mixed-precision training budget evidence

Read the [review report](../../docs/experiments/20261026_Mixed_Precision_Muon_and_Parallel_GPU_Budget_Review.md)
for the corrected 3090 Ti unlock/TITAN comparison, full-scope memory plans,
MiniCPM/Spark estimates and separate parallel-preparation/joint-pipeline schedules.

- [Assumptions](../../configs/mixed_precision_training_budget_v1.json): explicit
  work volume, model envelopes, unmeasured hardware delivery, optimizer routing,
  memory/communication allowances and campaign reserve.
- [Estimate](estimate.json): exact numbers behind the report, including 100M,
  1B and 5B positions, 8K context sensitivity, slower fallbacks and M2 what-ifs.
- [Input manifest](input_manifest.json): seven new frozen snapshots and fifteen
  reused snapshots from the first review, each SHA256/size checked.
- [External sources](external_sources.json): primary hardware, optimizer, kernel
  and pricing sources checked on 26 October 2026.
- [Validation](validation.json): nine passed offline accounting/evidence checks.

Reproduce offline into a **new** output path:

```bash
.venv/bin/python scripts/estimate_mixed_precision_training_budget.py \
  --output reports/local/mixed-precision-budget-replay.json
```

The script imports only the Python standard library. It verifies evidence before
calculating and refuses to overwrite an output file. It loads no model tensors,
accesses no GPU or network, and changes no training settings. The snapshots of
trainer/optimizer source are evidence text and are never imported or executed.

The M2 BF16 executable budget is deliberately null; conditional hours are shown
separately. CUDA rates are assumptions and unlock ratios come from external GEMM
measurements. There is no local NVIDIA benchmark or convergence-equivalence claim.
All proposed encoder and intrinsic-module training remains included. Existing
ignored weights, checkpoint payloads and cached run directories remain outside Git.

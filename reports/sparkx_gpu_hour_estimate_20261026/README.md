# Spark full-training budget review

26 October 2026, America/Chicago. Read-only review and offline arithmetic;
no native GPU process, paid resource, model download or media acquisition.

- [Full review](../../docs/experiments/20261026_SparkX_Formal_Full_Finetuning_Compute_Estimate.md)
- [Explicit assumptions](../../configs/sparkx_formal_training_budget_estimate_v1.json)
- [Deterministic estimate](estimate.json)
- [Input snapshot manifest](input_manifest.json)
- [Official specifications/prices/support](external_sources.json)
- [Offline verification](validation.json)

For 1B total charged positions, one H200 BF16-compute schedule is estimated at
116–417 GPU-hours; three schedule equivalents plus initial qualification are
355–1,266 GPU-hours, about $1,590–$5,660 at the checked rate. These are explicit
planning assumptions, not measured Spark/CUDA performance or convergence bounds.
The current full FP32 Mac configuration cannot fit; its numerical time projection
is marked compute-only and non-executable. Each dual-card pair is a separate
sharded/offloaded job, and occupied GPU-hours differ from elapsed hours.

The 15 small input snapshots preserve the actual working-tree documents and
recorded MiniCPM evidence used. The newer sustained run has 12 updates, ten
steady updates, 0.11874 steady updates/s and 52.246 processed positions/s.
All 381 original tensors changed; the source report's unqualified checkpoint
reload flag is retained. No result is promoted.

Reproduce without importing a model or using the network:

```bash
.venv/bin/python scripts/estimate_sparkx_training_budget.py \
  --output reports/local/sparkx-budget-recomputed.json
```

Nine offline checks independently verify snapshot hashes, original phase means,
changed-tensor evidence, Decimal hour/cost arithmetic, two-GPU occupancy,
independent-task accounting, data-volume scaling, infeasible Mac state,
attention assumptions, local report links and byte-identical CLI replay.
Source documents and other concurrent work remain untouched.

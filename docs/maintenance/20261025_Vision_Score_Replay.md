# Vision-curriculum score validation and historical replay

25 October 2026. `replica.vision_curriculum.score_report.v1` has two explicitly
selected validation routes. New reports never use the historical overlay.

## Current v1

Use `--validation-version current-v1`. The score must already contain the exact
`reload_report_sha256`; the current run, reload, protocol and score validators
then run without an overlay. A supplemental lineage or outer-guard overlay is
rejected.

## Historical v1 with supplemental lineage

Use `--validation-version historical-v1-supplemental-lineage` only for archived
reports that lack `reload_report_sha256`. The route requires all of:

1. Original score, protocol, run, reload, per-arm outer guard and
   `supplemental_lineage.json` byte hashes.
2. Matching arm, protocol, run, data manifest and score bindings in the original
   supplemental lineage.
3. The original 27-pin protocol inventory. Its omitted
   `scripts/run_capacity_preflight.py` pin is supplied only in memory from the
   lineage's `native_launch.guard_sha256`; current structural validation then
   runs. The original protocol and run retain their 27-pin identities.
4. A per-arm outer guard whose command, output, protocol, successful status,
   resource policy and peak driver reading match the run. The archived internal
   guard sampled a lower peak than the outer guard; only the in-memory run view
   uses the outer peak for the later cost-consistency check.
5. For archived reloads without `payload_sha256`, derive that self-hash from the
   unchanged payload in memory. Finally add the authenticated reload file hash
   to the in-memory score copy and run the unchanged strict score validator.

No input is rewritten. The fresh receipt records every overlay, original hashes,
engineering result and unchanged learned-quality disposition. Historical
normalization is not native execution and is not a new quality evaluation.

## Verified archived results

The following completed successfully on 25 October 2026:

```bash
.venv/bin/python scripts/replay_vision_curriculum_score.py \
  --validation-version historical-v1-supplemental-lineage \
  --score-report reports/vision_curriculum_20261025/adamw_score01/score_report.json \
  --protocol data/vision_curriculum_20261025/protocol.json \
  --run-report runs/vision_curriculum_20261025/adamw01/run_report.json \
  --reload-report runs/vision_curriculum_20261025/adamw01_reload/reload_report.json \
  --outer-guard-report reports/vision_curriculum_20261025/adamw01.guard.json \
  --supplemental-lineage reports/vision_curriculum_20261025/supplemental_lineage.json \
  --output reports/vision_score_replay_20261025/adamw_validation.json
```

The Muon command substitutes `muon_score04b/score_report.json`, `muon04`,
`muon_reload04b/reload_report.json`, `muon04.guard.json` and
`muon_validation.json`. Receipts:

- `reports/vision_score_replay_20261025/adamw_validation.json`
- `reports/vision_score_replay_20261025/muon_validation.json`

Both reports validate their engineering evidence and retain
`learned_quality_pass=false`. Original scores and supplemental lineage are
unchanged. See [test provisioning](20261025_Test_Provisioning.md) for ignored
asset requirements and the distinction between unavailable integration coverage
and passing coverage.

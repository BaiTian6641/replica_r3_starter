# E3 DAgger aggregation: initial 2/5, after 0/5

23 October 2026 UTC. Protocol `configs/e3_dagger_v1.json` committed before
execution. Fresh four-train/one-dev/four-held-out family; initial accepted E3
extension; modules-only training; host teacher labels computed on model-visited
states; 8 updates × 4 examples (26 aggregated examples); no base save.

**Result: initial held-out/dev 2/5; after aggregation 0/5.** Training loss fell
to 0.023, but the post-training model emitted malformed/unauthorized tool
proposals and completed no episode. The aggregate includes two successful train
rollouts before labeling. This is a negative DAgger result, not a policy
promotion.

| Stage | Completions | Notes |
|---|---:|---|
| Initial E3 extension | **2/5** | 2 held-out completions; 1 false success claim in another case |
| After 8 DAgger updates | **0/5** | 7 decisions each, no evaluator success |
| Rollout examples | 26 | from 4 train cases; 2 train episodes succeeded |
| Training loss | 0.245 → 0.023 | teacher-forced |
| Peak driver | 16.21 GiB | 430 s; extension 37.6 MB, no base save |

No candidate was published; the exported extension is diagnostic only. The
initial 2/5 is encouraging but not a generalization claim: the family is fresh,
small and the initial policy still made a false success claim. DAgger as
implemented did not improve held-out selection.
Evidence: [report](../../reports/e3_dagger_attempt01/run_report.json).

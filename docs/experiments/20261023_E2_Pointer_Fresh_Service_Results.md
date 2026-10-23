# E2 pointer on fresh values: 3/12, partly truncated copies

23 October 2026 UTC. Consumer-only evaluation of the published pointer
extension on a new 12-case family; no training, checkpoint or default change.

## Result

| Metric | Value |
|---|---:|
| Fresh held-out exact | **3/12 (25%)** |
| Pointer provenance complete | 12/12 |
| Exact-source fallback | 12/12 |
| Training / optimizer updates | none |

Hits: `magenta`, `turquoise`, `lavender` (correct records). Several misses were
partial subword copies (`saff` for `saffron`, `ilion` for `vermilion`, `cer` for
`cerulean`), and other pointer choices came from distractor records
(`same_attr`/`same_entity`). The pointer is source-dependent but does not
reliably copy complete novel values across sources; exact text remains the
default.

## Costs

**7.1 s**, **11.0 GiB** sampled peak driver, one model load, no optimizer or
checkpoint. Raw:
[report](../../reports/e2_pointer_fresh_service_attempt01/run_report.json).

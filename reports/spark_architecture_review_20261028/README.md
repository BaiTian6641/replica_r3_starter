# Spark-X2.6-4B architecture review evidence

28 October 2026, America/Chicago. See the [full review](../../docs/architecture/27_Spark_X2_6_4B_Architecture_Research_Review.md).

This package records fresh primary-source research and a document/code cross-check. It downloads only bounded public text/metadata, inspects existing safetensors headers without loading weights, and runs small CPU library checks. It changes no model source pin and performs no native model execution or training.

| File | Purpose |
|---|---|
| `source_manifest.json` | URLs, retrieval timestamps, byte lengths and SHA-256 values for 62 source snapshots, including six license/attribution files |
| `sources/` | Exact public config, code, template, card, Hub metadata, backend source and benchmark SVG text snapshots; upstream licensing applies |
| `architecture_audit.json` | Local input hashes, source comparisons, checkpoint-header inventory, parameter/cache arithmetic, 30 vendor benchmark rows and eight existing receipt checks |
| `cache_template_probe.json` | CPU cache backing-storage behavior and template-mode validation; torch/Transformers versions |
| `literature_review.json` | Primary papers, inspected versions, search scope and limits of transfer to Spark |
| `collect_public_sources.py` | Optional network collection of text sources, capped at 2 MiB per response; never downloads weights |
| `audit_architecture.py` | Offline stdlib reproduction of header/source/receipt checks and arithmetic |
| `probe_cache_and_template.py` | CPU library checks; never imports or instantiates Spark model code |

Commands run from the repository root:

```bash
python3 reports/spark_architecture_review_20261028/audit_architecture.py
.venv-spark-bf16/bin/python reports/spark_architecture_review_20261028/probe_cache_and_template.py
```

Both completed successfully. The audit uses locally provisioned, ignored source checkpoints and old run receipts. Missing provisioned files must be reported as missing; they must not silently become a successful check. The source collector is optional and rewrites this review's snapshot manifest; prefer the retained evidence for reproducing this dated review. Model and backend artifact URLs are immutable. The three GitHub latest-commit metadata lookups record discovery context and can change if recollected; they do not select or replace the hard-coded inspected backend revisions.

Upstream source snapshots preserve exact bytes, including original trailing whitespace. The unrestricted staged whitespace check reports those upstream lines; the check excluding `sources/` passes for authored documents, scripts and receipts. Do not normalize the snapshots: that would invalidate their public-source hashes. The staged artifact guard passes. The original Apache licenses and plugin notices are retained beside the snapshots.

The current repository contains unrelated uncommitted work. This review owns only this evidence directory and the new Doc27. Earlier hashed architecture indexes and reports are preserved. Source-payload identity comparisons use Hub LFS metadata and prior acquisition receipts; this review does not rehash local multi-gigabyte weight payloads. All context/resource figures are analytical, except the explicitly labeled small CPU cache probe and previously recorded native smoke measurements.

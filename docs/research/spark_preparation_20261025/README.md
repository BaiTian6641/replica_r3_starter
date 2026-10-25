# Spark-X2.6 FP8/BF16 source preparation — 25 October 2026

**Status: preparation only. No weight payload was downloaded, no remote Python was
executed and no native forward, backward, optimizer step or reload was run.**
This package does not qualify MPS FP8 training or any native Spark execution.

The reusable implementation is `src/replica_lab/spark_preparation.py`; the CLI is
`scripts/prepare_spark_sources.py`; the validated plan is
`configs/spark_source_preparation_v1.json`; tests are
`tests/test_spark_preparation.py`. Evidence from the run is under
`reports/spark_preparation_20261025/run_01/`; the machine manifest is
`reports/spark_preparation_20261025/run_01/preparation_manifest.json`
(fingerprint `1142605a342ca0ec96548bb75553a92a073f71f50a1020f893664e02ec1427b3`).

## Official identities

| Role | Repository | Revision | Storage |
|---|---|---|---|
| Requested adaptation base | `XHToken/Spark-X2.6-4B-FP8` | `0318d6f6ab070c7b5b43052db571489b2508006c` | compressed-tensors float-quantized FP8 weights, BF16 remainder |
| Independent control | `XHToken/Spark-X2.6-4B` | `66e3ae7bee54e0a523d692cb8ef841f35409807b` | BF16 safetensors |

Both cards declare Apache-2.0. The FP8 card names the BF16 release as parent, so
they are related releases rather than bitwise-equivalent copies. They are
distinct repositories with distinct revisions and disjoint weight-shard LFS
digests. Many small assets (tokenizer, chat template, custom Python, LICENSE,
generation config) have identical bytes; those are recorded as shared bytes, not
as shared weight evidence. Neither artifact is a dequantized substitute for the
other.

## What was actually inspected

* 39 bounded HTTP requests, 29,708,033 bytes total, against a 64 MiB plan ceiling.
  Every safetensors header used byte ranges; an HTTP 200 to a range request is
  refused before the body is read.
* FP8 `model.safetensors`: 4,451,626,120 bytes, a 55,424-byte header
  (`18d82da0cb0276bfc3e1d3fc9655d7f0c71ffbd021bd5da5c2261f7fd416840a`), 470
  tensors. The header's tensor payload covers the shard exactly.
* BF16 five shards: 8,224,192,408 bytes together, five headers, 290 tensors.
  The published index (`total_parameters=4,112,079,360`,
  `total_size=8,224,158,720`) matches the header inventory.
* FP8: 180 F8_E4M3 projection tensors storing 3,774,873,600 elements and 180
  BF16 channel scales storing 1,142,784 elements. All remaining 337,205,760
  parameters are BF16. No zero-point or quantized-KV tensors are present.
* No `lm_head.weight` is serialized in either source. The 131,072 × 2,560
  embedding is the tied logits matrix.
* `config.json`, tokenizer config, chat template, generation config, LICENSE,
  README and both custom Python files were fetched and hashed. Custom Python was
  parsed with `ast`; it was never imported. Its import-time calls are limited to
  `logging.get_logger(__name__)` and `ALL_LAYERNORM_LAYERS.append(Spark2_5RMSNorm)`.
* Hub metadata counts agree with the header-derived counts but are not treated as
  tensor evidence.

Exact URLs, revision, byte counts, SHA256 values, HTTP status, `Content-Range`,
ETag, redirect target and refusal evidence are in the manifest's `provenance`
records. Retained text assets and raw safetensors headers are in
`run_01/assets/` and `run_01/headers/`; the tokenizer/vocab/merges bundles were
hashed during the bounded fetch and deliberately not retained (they are
reproducible inputs, not weights). A derived
`reports/spark_preparation_20261025/header_bundle.json` packages the raw headers
for local-header intake; it is not part of the CLI-created run directory.

## Documents

1. [Source and tensor audit](01_Source_and_Tensor_Audit.md): pins, hashes,
   code review, tensor accounting.
2. [Precision capability and qualification protocol](02_Precision_Capability_and_Protocol.md):
   independent FP8/BF16 matrix, compressed-tensors semantics, isolated
   environment, matched inference/gradient/update/reload protocol.
3. [Adapter ABI](03_Adapter_ABI.md): dedicated `spark2_5` adapter design; this is
   a design in the manifest, not a disconnected fake implementation.

## Reproduce

Online, into a path that does not exist:

```bash
python3 scripts/prepare_spark_sources.py \
  --output reports/spark_preparation_20261025/run_02
```

Reuse captured headers while still fetching the text assets (useful when range
access is unavailable):

```bash
python3 scripts/prepare_spark_sources.py \
  --output reports/spark_preparation_20261025/run_local_headers \
  --header-bundle reports/spark_preparation_20261025/header_bundle.json
```

Fully offline from previously captured headers and assets (all assets, including
the large tokenizer files, must be present):

```bash
python3 scripts/prepare_spark_sources.py \
  --output reports/spark_preparation_20261025/run_offline \
  --offline \
  --header-bundle /path/to/headers.json \
  --assets-from /path/to/assets
```

The CLI refuses an existing output directory, checks free disk before creating
it, enforces per-asset, per-header and cumulative byte ceilings, rejects
unpinned revisions, weight payloads in the asset list, aliased source
repositories/revisions, missing or malformed shard headers, overlapping/gapped
tensor offsets, mismatched parameter accounting and any change to pinned SHA256
values. Failures write `failure.json` with accumulated fetch/refusal records.

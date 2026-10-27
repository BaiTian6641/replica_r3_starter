# Spark dataset preparation: metadata replay and remaining intake work

**26 October 2026 · Metadata/documentation status for the SparkX 2.6 4B dataset review · Not an ingestion manifest, asset-QA pass or training admission.**

This note closes the documentation/replayability gap in the dataset research package. It adds a standard-library [verifier](../../scripts/verify_sparkx_dataset_research.py), a fresh [validation report](../../reports/sparkx_dataset_validation_20261026/validation.json), a prospective [intake record](../../reports/sparkx_dataset_validation_20261026/intake_manifest.json), and offline [tests](../../tests/test_sparkx_dataset_validation.py). The existing [catalog](../datasets/20261026_SparkX_Dataset_Catalog.json), [plan](../datasets/20261026_SparkX_Dataset_Plan.md), [research report](../../reports/sparkx_dataset_hunt_20261026/README.md), [MiniCPM validation plan](../datasets/20261026_MiniCPM_Dataset_Validation.md) and runtime [registry](../../configs/dataset_registry.json) were not edited.

**No native execution or downloads occurred.** No dataset loader or model was run. No image, audio or video asset, model weight or bulk annotation collection was downloaded. The runtime registry remains unchanged and quarantined. The only prior asset access was the documented retrieval of three small Clotho provenance CSVs (plus license/source text) by the original research task; this task did not repeat it.

## 1. What is now replayable

The verifier reads local catalog/evidence documents only. It does not use the network, media bytes, dataset loaders, models or credentials. It recomputes and checks:

| Area | Recomputed/validated |
|---|---|
| Inventory | 36 source families; 40 catalog repository pins plus 2 documented lookups = 42 inspected Hugging Face repositories; 125 evidence records |
| Hash coverage | 121 records with well-formed SHA-256 digests and positive byte counts; exactly 4 unresolved requests, each with its documented HTTP error |
| Source pins | 40-hex revision pins; pinned card URLs; API/card evidence URLs; API evidence digest equals the metadata-snapshot digest; 38 resolved catalog cards, 2 documented 404 catalog cards, and 2 lookup cards (one resolved, one 404) |
| Split metadata | Every reported config/split has a feature schema, positive example counts and numeric sizes; 23 annotation-only or card-poor repositories have explicitly no reported split metadata, recorded as an absence rather than a validated layout |
| Rights/provenance requirements | Non-empty rights review, annotation provenance, split-group keys and at least two source-specific intake checks per family; restricted/mirror priorities, noncommercial/other licenses, gated repositories and evaluation-only roles cannot receive P1 priority or admission |
| Clotho audit | 3,839 / 1,045 / 1,045 records (5,929 total), unique filenames equal row counts, license counts sum to rows, development 1,577 CC0 and 1,572 CC-BY |
| Claims | Catalog and evidence declare `training_admitted=false`, `auto_admit=false`, `media_assets_downloaded=0`, `model_runs=0`, bulk annotation downloads 0 and three provenance CSVs; every family is unaudited and unadmitted |
| Registry | `configs/dataset_registry.json` matches its pinned SHA-256, remains `research_catalog_not_an_ingestion_manifest`, all 13 entries quarantined/not downloaded/`auto_admit=false` |
| Documents | README/plan count and no-asset/no-training claims agree with computed values; local markdown links resolve |
| Historical inputs | All 15 input files are hashed; the frozen architecture digest matches its pin; snapshot drift is reported for mutable documents |

The report records 44 checks. A passing verdict means only that the documented metadata claims are internally consistent and reproducible. It does **not** mean the datasets are licensed, de-duplicated, split-safe, high quality or trainable.

The earlier [documentation validation](../../reports/sparkx_dataset_hunt_20261026/documentation_validation.json) is retained as an input and checked only for internal arithmetic consistency (426 checks, 426 passed, 0 failed). Its generator/verifier was not retained, so it is not treated as clean-export replay evidence and its counts are not re-asserted here as an independent pass.

## 2. Reproduce

From the repository root:

```bash
python3 scripts/verify_sparkx_dataset_research.py
python3 scripts/verify_sparkx_dataset_research.py
shasum -a 256 reports/sparkx_dataset_validation_20261026/validation.json \
  reports/sparkx_dataset_validation_20261026/intake_manifest.json
.venv/bin/python -m pytest tests/test_sparkx_dataset_validation.py -q
```

The two verifier runs produce byte-identical outputs: the report contains input SHA-256 digests and fixed review-date metadata, but no wall-clock timestamp, temporary path or random identifier. `--check-only` runs the checks without writing files. Exit status is `0` for `pass_metadata_only`, `1` for `contradiction_refused` (a refusal report is written, but no intake record), and `2` when an input is missing or structurally unreadable (no outputs are written).

The tests copy the input package to a temporary directory and exercise behavior: valid replay and determinism, count mismatch, unresolved/contradictory evidence, tampered registry and hash pins, missing rights/split/check fields, relative provenance paths, and absent inputs. They run offline.

The tests require the local research files listed in `INPUT_ROLES`. A clean export
without those files marks an explicit provisioning skip; the verifier itself
exits with code 2 for missing inputs and never reports a pass. The successful
18-test run used the complete local research package.

## 3. Outputs and their meaning

The [validation report](../../reports/sparkx_dataset_validation_20261026/validation.json) contains the `pass_metadata_only` verdict, input hashes, exact inventory, the four unresolved records, all checks and failures, source-pin drift, warnings, and limitations. Its `verdict.registry_unchanged` is `true`; `verdict.training_admitted` and `verdict.asset_qa_run` are `false`.

The [intake manifest](../../reports/sparkx_dataset_validation_20261026/intake_manifest.json) is a prospective planning record, explicitly marked:

- `metadata_only_not_an_ingestion_manifest: true`
- `training_admitted: false`
- `media_assets_downloaded: 0`
- `asset_qa_not_run: true`
- `registry_unchanged: true`
- `native_execution_not_run: true`

It lists a proposed **25 GiB** asset ceiling with status `prospective_not_authorization`, a free-space check of `not_run`, and the separate 50 GiB native RAM guard. Its `next_bounded_asset_audit` has `authorized: false`, `status: not_started`, and records the prerequisites: explicit named authorization, resolution/closure of the four unresolved evidence requests, a per-source rights/source map, free-space and separate cache/checkpoint accounting, an immutable media manifest with independent byte hashes, executed duplicate/source-overlap checks, a bounded ~100-source-group QA sample per selected source, a separate asset-QA report, and separate registry admission. The P1 families listed are candidate review order only, not selection or clearance.

## 4. Completed metadata research versus missing work

| Stage | Status | Evidence/limit |
|---|---|---|
| Source inventory, revisions and card/API pins | **Completed** | 36 families, 42 repositories, 125 evidence records; 121 hashed, 4 unresolved |
| Documentation replay (inventory, counts, references, pins, claims) | **Completed** | `validation.json`, 44 checks, deterministic |
| Clotho provenance CSV count audit | **Completed** | 3 CSV records, 5,929 rows; not audio QA |
| Rights resolution / clearance | **Not started** | 0 of 36 families cleared; publisher statements only |
| Duplicate/source-overlap graph execution | **Not started** | Lineages documented; no exact-media or perceptual matching executed |
| License compatibility review | **Not started** | No redistribution, intended-use or gate acceptance decision |
| Asset acquisition | **Not started** | 0 media assets; 0 bulk annotations; 25 GiB ceiling merely prospective |
| Asset-level QA | **Not run** | 0 source groups reviewed; no waveform/frame/label inspection |
| Split materialization and enforcement | **Not started** | Group keys are requirements, not a validated split graph; 23 repositories expose no split metadata |
| Runtime registry admission | **Not started** | Registry unchanged, 13 quarantined entries, 0 admitted |
| Native execution / fine-tuning / learning experiments | **Not run** | No model runs, dataset-loader runs or training |
| Dataset quality or capability claims | **Not established** | Metadata consistency is not quality evidence |

The research artifacts support planning a bounded asset audit. They do not support a claim that any family is ready for training, and the manifest must not be cited as an ingestion manifest or asset-QA record.

## 5. Precise non-claims

- No dataset, subset, mirror or media asset was admitted, cleared or downloaded by this task.
- No asset QA, duplicate detection, split-safety audit, rights review or gate acceptance was performed.
- No native execution, dataset-loader run, model training or MiniCPM pilot occurred.
- No publisher row count was converted into a unique, licensed or training-ready asset count.
- The 25 GiB ceiling, ~100-source-group sample and P1 review order are prospective design choices, not authorization.
- The registry hash and quarantine policy were checked read-only; no registry entry was added, promoted or modified.
- A `pass_metadata_only` verdict is not a dataset-quality, legal, safety or admission verdict.

## 6. Files and next step

Added by this task:

- `scripts/verify_sparkx_dataset_research.py` — offline verifier and intake-record builder.
- `tests/test_sparkx_dataset_validation.py` — focused behavioral tests.
- `reports/sparkx_dataset_validation_20261026/validation.json` — generated validation report.
- `reports/sparkx_dataset_validation_20261026/intake_manifest.json` — prospective, non-ingestion record.
- `docs/experiments/20261026_Spark_Dataset_Preparation.md` — this note.

The existing repository snapshot records that the acquisition guide `docs/08_Dataset_Acquisition_and_Curriculum.md` has changed since its historical pin. The verifier reports this as a warning and does not repair or re-certify the changed document; the runtime registry and the other pinned research documents still match. The next permissible step is an explicitly authorized bounded asset audit under a new manifest, after the prerequisites in the intake record are satisfied.

## 7. Hashed clean-checkout provisioning

This revision adds a manifest-based provisioning path for the existing research inputs:

- `scripts/prepare_sparkx_research_bundle.py` — hashes every required input, records its repository-relative role and byte count, and verifies git tracking.
- `reports/sparkx_first_tranche_20261026/research_bundle_manifest.json` — the generated manifest and bundle digest.

Manifest entry paths are repository-relative; absolute home, temporary or `/private/tmp` paths are rejected in those entries. Historical receipt contents may quote the origin workspace and are not rewritten. A manifest input that is missing, size-mismatched, hash-mismatched or untracked fails verification. Untracked files are refused by default even when they exist locally, because a clean checkout would not contain them.

```bash
# create/refresh the manifest after the listed inputs are committed
python3 scripts/prepare_sparkx_research_bundle.py --write

# clean-checkout verification
python3 scripts/prepare_sparkx_research_bundle.py --check

# development-only override; records clean_checkout_ready=false
python3 scripts/prepare_sparkx_research_bundle.py --check --allow-untracked
```

The named provisioning command for tests and operators is:

```bash
python3 scripts/prepare_sparkx_research_bundle.py --write
```

Every listed input is committed, and the regenerated manifest reports `clean_checkout_ready: true`; `--check` passes without `--allow-untracked`. A checkout missing any listed input must use the named provisioning command before claiming replay.

## 8. Bounded first tranche protocol

`configs/sparkx_first_tranche_v1.json` defines a **protocol only**, with `download_authorized: false`, `bulk_download: false`, `native_execution: false` and `training_admitted: false`. `scripts/prepare_sparkx_first_tranche.py` applies it to a JSONL asset manifest and exact asset bytes.

The protocol pins four research candidates, without acquiring them:

| Source | Pinned revision | Card license | Gate |
|---|---|---|---|
| `allenai/pixmo-points` | `2b5c6931e790e00ae00d4a2857e5f95d88f09a66` | ODC-BY | not acquired or cleared |
| `allenai/pixmo-cap` | `edce6390d9d5be6c8db0d863fbe62718c88988a4` | ODC-BY | not acquired or cleared |
| `Fhrozen/FSD50k` | `ccf1acaff12f3f4a4c10052dddafbb6d22152c9b` | CC-BY-4.0 | not acquired or cleared |
| `openslr/librispeech_asr` | `71cacbfb7e2354c4226d01e70d77d5fca3d04ba1` | CC-BY-4.0 | not acquired or cleared |

Because no external source has a clearing decision, every real-source asset is refused with `source_not_acquired_or_cleared`. The only admitted evidence in this task is the repository-owned synthetic fixture set under `reports/sparkx_first_tranche_20261026/synthetic_assets/`. Those JSON files are labelled `owned_synthetic_not_external_media`; they are not images, recordings or videos from any external dataset and carry no external license.

Admission rules, all deterministic:

- **Fields and provenance:** every required field (asset id, source id, pinned revision, group key, SHA-256, bytes, media kind, license, path, provenance, synthetic flag) must be present and non-empty.
- **Hash and size:** SHA-256 and size must match the exact bytes. Paths must remain inside the declared asset root; absolute paths and `..` are rejected.
- **License:** only an explicit allowlist (`odc-by`, `cc-by-4.0`, `cc-by-sa-4.0`, `repository_owned_synthetic`) is accepted. NC/ND/ODbL/unknown/`other` and evaluation-only or gated sources are refused.
- **Dedup:** exact SHA-256 duplicates are rejected, with the first asset id recorded. Perceptual or near-duplicate detection is **not** claimed.
- **Leakage:** a group key maps to exactly one split and one source; an explicit `declared_split` must equal the computed split. Any conflict triggers the global leakage stop and refuses the whole tranche.
- **Roles:** train/development/audit is a deterministic 70/15/15 function of `sha256(salt || group_key)`, so a source group cannot straddle splits. The committed synthetic fixture resolves 2/2/2.
- **Caps and stops:** per-asset, per-source and tranche asset/byte caps stop admission and are recorded in the receipt. Source integrity failures (revision, license, provenance, hash, missing file) stop that source. A duplicate rate above 0.2 stops the source. Fewer than three admitted assets yields `refused_no_admission`.

Outputs are `tranche_receipt.json`, `admitted_assets.jsonl` and `rejected_assets.jsonl` in `reports/sparkx_first_tranche_20261026/`. The receipt states `synthetic_only: true`, `external_media_acquired: false`, `training_admitted: false`, `network_used: false` and `native_execution: false`.

## 9. Tests and precise non-claims

```bash
.venv/bin/python -m pytest tests/test_sparkx_first_tranche.py -q
.venv/bin/python -m pytest tests/test_sparkx_residency.py -q
```

The self-contained tests build temporary trees and owned synthetic bytes. They cover manifest generation and hashing, clean-checkout refusal of untracked inputs, hash/traversal/digest tampering, balanced role assignment, exact dedup, duplicate-rate and integrity stops, leakage refusal, cap stops and configuration rejection. The artifact-dependent manifest/receipt tests use `require_local_artifacts`; in a clean export they skip with the named `--write` provisioning command, while a provisioned workspace runs them.

This revision does **not** claim:

- any external dataset, media asset, annotation or license was acquired, cleared, audited or admitted for training;
- any asset-level QA, perceptual dedup, rights clearance or split-safety validation beyond the documented group-key checks;
- any dataset-loader, model, MPS or native execution;
- that synthetic fixtures are external media or that a 2/2/2 protocol fixture establishes quality or scale.

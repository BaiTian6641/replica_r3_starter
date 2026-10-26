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

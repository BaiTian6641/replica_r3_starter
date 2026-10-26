# Real-image evidence tranche

**26 October 2026 · attempt03 selected · rights, content and offline QA verified.**
Seven bounded real-image derivatives are admitted for **local image-input
validation only**. `training_admitted` remains `false`; this is not a training
campaign, a native model evaluation or evidence of real-image competence.

The frozen [protocol](../../configs/real_image_tranche_v1.json) was committed
before asset-byte acquisition at `59d98c6` (SHA-256
`e18ce054aeaa9a9bdfa3d3a2cd0c13971927c71465bfb959a1236f82ae2e6714`).
The selected [manifest](../../reports/real_image_tranche_20261026/attempt03/manifest.json)
binds implementation `567abbda5dd345e753ab3f507f490c4323ee1c6712655607d3138539b01bc358`,
from `576589b` plus the integrity fix at `1571e4e`.

## Admitted assets

All source rights and provenance are recorded per asset in the manifest. The
intended rights jurisdiction is the United States; this is a provenance review,
not legal advice. Original media were not downloaded. Each file is the declared
Wikimedia 960-pixel-wide JPEG derivative, pinned by source page, current
revision, original SHA-1 metadata and independent derivative SHA-256.

| Asset | Source / author | Rights | Split | Decoded geometry | Bytes | SHA-256 prefix |
|---|---|---|---|---:|---:|---|
| Apollo 17 Earth | [NASA AS17-148-22727](https://images.nasa.gov/details/as17-148-22727), Apollo 17 crew | Public domain | train | 960×961 | 267,290 | `16eaa54446f7` |
| Apollo 11 flag | [NASA AS11-40-5874](https://images.nasa.gov/details/as11-40-5874), Neil Armstrong | Public domain | train | 960×969 | 296,954 | `deecff91f88e` |
| Migrant Mother | [Library of Congress](https://www.loc.gov/pictures/item/2017762891/), Dorothea Lange / FSA | Public domain; LoC no known restrictions | audit | 960×1248 | 255,383 | `95b09023d747` |
| Columns of the Temple of Neptune | [Met object 438949](https://www.metmuseum.org/art/collection/search/438949), Constantin Hansen | CC0 1.0 | audit | 960×1202 | 307,422 | `df0c3df3caf7` |
| LBJ presidential oath | [NARA NAID 194235](https://catalog.archives.gov/id/194235), Cecil Stoughton / White House | Public domain, U.S. federal work | development | 960×747 | 164,851 | `0245d5a08ea4` |
| The Night Watch | [Rijksmuseum](https://hdl.handle.net/10934/RM0001.COLLECT.5216), Rembrandt (d. 1669) | Public domain | train | 960×781 | 181,490 | `54d8ad29b61f` |
| Mameyes landslide | [USGS](https://www.usgs.gov/media/images/catastrophic-landslide-0), Randall Jibson | Public domain | train | 960×646 | 261,593 | `833c2ead80be` |

The six source groups are NASA Apollo, LoC FSA/OWI, Rijksmuseum, Met Open
Access, USGS and the U.S. National Archives. The frozen group-hash rule assigns
**4 train / 1 development / 2 audit** assets with no group crossing a split.
The initial six-candidate set lacked a development group; the distinct NARA
collection was added without changing the frozen salt or existing group names.
No benchmark source ID is admitted. This is explicit source/group exclusion,
not exhaustive semantic decontamination.

## Acquisition and verification

- Acquisition made **7 logical asset requests**, all direct HTTP 200 targets,
  with zero failures and **1,734,983 bytes** received. No original URL was
  fetched. The per-asset cap was 2 MiB and total cap 8 MiB.
- All seven decode fully as JPEG with width 960 and source-consistent aspect
  ratios. Portrait heights exceed 960 where applicable; `960px-` constrains
  width, not maximum dimension.
- Exact SHA-256 values are unique across all seven assets. No perceptual or
  near-duplicate detection is claimed.
- Offline verification recomputed every hash, byte size, decode, geometry,
  split, rights field and source exclusion with **zero network requests**.
  The selected verification reports **41/41 checks true**, no failures and no
  warnings, with `implementation_matches_manifest=true`.
- `acquisition.json`, `verification.json` and `report.json` preserve the two
  phases separately. Missing or incompatible acquisition history now fails
  closed while asset QA still runs.
- The selected directory contains 11 files totaling **1,862,449 bytes**. The
  independent review adds 44,961 bytes.

Commands:

```bash
.venv-vision/bin/python scripts/prepare_real_image_tranche.py \
  --config configs/real_image_tranche_v1.json \
  --assets-dir reports/real_image_tranche_20261026/attempt03/assets \
  --report-dir reports/real_image_tranche_20261026/attempt03 --download

.venv-vision/bin/python scripts/prepare_real_image_tranche.py \
  --config configs/real_image_tranche_v1.json \
  --assets-dir reports/real_image_tranche_20261026/attempt03/assets \
  --report-dir reports/real_image_tranche_20261026/attempt03 --verify
```

The Pillow-enabled suite passed **14 tests with one intentional complementary
skip**. The base environment passed **8 tests with seven decode-only skips**,
exercising the Pillow-absent fail-closed branch. Tests were run by the parent,
not claimed by the independent reviewer.

## Independent review and limitations

The [independent review](../../reports/real_image_tranche_20261026/independent_review.json)
(SHA-256 `45cbd115ae32cc938839e1f6682572a615af988f1edfbfda0f90bb7274ad0271`)
returns `verified_with_declared_limits`. It independently checked hashes,
geometry, splits, dedup, receipt bindings, current source revisions, original
SHA-1 metadata, institutional rights/provenance and actual image content. All
seven images match their declared subjects.

Two informational limits remain. Implementation drift is a visible warning,
not by itself invalidation of immutable verified bytes; attempt03 has no such
drift. Request counts are logical asset fetches, not redirect hops; the
selected URLs were direct HTTP 200 targets. Attempt01/02 assets are byte-
identical to attempt03 and remain local as superseded evidence. Attempt01's
old verifier overwrote its acquisition report; attempt02 introduced stable
phase receipts; attempt03 adds the fail-closed missing-history guard and full
verification metadata.

No audio/video tranche is admitted. No broad real-image quality claim follows
from seven assets. Current MiniCPM5 vision learnability remains a separate
open gate, so these files must not be silently promoted into a training run.

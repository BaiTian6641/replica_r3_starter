# Audiovisual experience research evidence

**26 October 2026.** This package records a documentation/research review. It contains no new model benchmark, native-device test or training result.

| Artifact | Meaning |
|---|---|
| [Evidence manifest](evidence.json) | Fourteen papers, source URLs/hashes, selected reading scopes, code pins and retrieval failures |
| [Repository snapshot](repository_snapshot.json) | Results and file hashes observed at commit `2468f16dfdf590104133efb041f30ec57e669a3a`, including uncommitted files |
| [Revision 3 snapshot](architecture_v3_snapshot.txt) | Exact previous canonical document before this revision; earlier uncommitted work is preserved |
| [Analytical costs](analytical_costs.json) | Tensor payload and projection arithmetic for the proposed design |
| [Documentation verification](documentation_validation.json) | Local links, configuration consistency, arithmetic and available-source hashes; not neural capability tests |
| [Verification script](verify_research.py) | Standard-library-only repeatable checks |

The research synthesis is in [the document package](../../docs/research/sparkx_av_experience_20261026/README.md). The new proposal config is [revision 4](../../configs/sparkx_2_6_4b_product_plan_v4.json).

Public full texts and selected implementation files were read from temporary scratch storage. They were not copied into the repository or executed. Source hashes identify the inspected content; temporary files are not a permanent delivery dependency. Three unsuccessful URL attempts are retained in the manifest rather than treated as reviewed sources. Two DeepSeek report mirrors match exactly, without establishing original-release authentication.

Run documentation checks from the repository with:

```sh
python3 reports/sparkx_av_experience_20261026/verify_research.py
```

The script writes `documentation_validation.json`. If temporary source copies have been removed, it reports those checks as unavailable rather than recreating or claiming to have checked them. Concurrent repository files are compared with the observation snapshot and reported separately; a later change does not retroactively invalidate or update an earlier measurement. The script imports no project or machine-learning modules and accesses no device or network.

The cost ledger counts the paired bank, a fixed metadata allowance, FP32 core/event state, optional fast matrices and one shared active phrase-row table. It excludes unspecified variable state, implementation overhead and all separately trained module/encoder weights. Its remaining logical quota is not a measured VRAM margin. The 7 MiB bank estimate must never be described as total multimodal memory or total training cost.

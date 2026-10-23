# E2 pointer candidate publication and rollback

23 October 2026 UTC. **CPU-only metadata publication.** The observed opt-in
pointer candidate was registered as an accepted neural candidate; a deliberate
harmful candidate was rejected; an opt-in manifest was published in a separate
registry and rolled back to the exact-text baseline.

## 1. Published scope

- Candidate: `e2-pointer-copy-1`; extension weights
  `359af59a…`; changed parameters `copy_head.*`; parent E2 manifest
  `088498bd…`; evaluation **passed** (dev 3/6, held-out 2/10, shuffled 0/16).
- Artifact registry: pointer candidate `accepted`; deliberately harmful
  duplicate with `evaluation=failed` **rejected**.
- Separate manifest registry:
  - baseline `e2_exact_text` accepted;
  - pointer `e2_pointer_opt_in` accepted as a child;
  - rollback restored the baseline (`97319970…` active).
- The pointer manifest declares `default=False`; exact text remains the runtime
  default. No model/action runtime was promoted.
- `MemorySession.pointer_answer` now returns a typed `PointerAnswer` containing
  the selected token IDs/text, record ID, record hash, span hash, all contributing
  source IDs, candidate index/count and producer. It is a host provenance record;
  it does not alter the pointer's 3/6 dev, 2/10 held-out result.

## 2. Limits

This is publication metadata and rollback, not a new behavioral evaluation.
The pointer’s low absolute held-out accuracy and flat native training curve
remain limitations. The manifest registry is separate from the production
active manifest; “opt-in” means a caller may select the pointer path explicitly,
not that it is the default.

## 3. Costs

No native model load, optimizer update or checkpoint. Evidence:
[report](../../reports/e2_pointer_publication_attempt04/report.json) and
[artifact catalog](../../reports/e2_pointer_publication_attempt04/artifact_catalog.json).

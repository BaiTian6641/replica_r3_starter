# E2 pointer top-record continuation: no within-record gain

23 October 2026 UTC. Parent pointer vs a 16-update copy-head continuation on
**top-lexical-record, whole-word candidates**. Evaluation used a new 12-case
untouched family. **Result: rejected; parent 8/12, candidate 8/12.**

## 1. Arm

- Parent: published pointer extension (`runs/e2_pointer_head_attempt02/...`).
- Trainable: `copy_head` only; base/E2 writer frozen.
- Training: existing `data/e2_copy` train cases; top lexical record; whole-word
  candidate spans; 16 updates × 4; Muon/AdamW lr 5e-3; target decoded answer
  span.
- Evaluation: new 12-case family, same top-record/word interface, no
  generation; exact copied span.
- Extension v2, 53.1 MiB; no base copy; 34.9 s; 11.0 GiB peak.

## 2. Result

| Arm | Fresh exact |
|---|---:|
| Parent pointer (top record) | **8/12** |
| Continued candidate | **8/12** |

The candidate selected **identical spans** to the parent. Training loss stayed
roughly **12.3–12.6**, with no learning-curve descent. Misses were entity-name
spans inside the relevant record (`Brook`, `Cliff`, `Field`, `Gale`), matching
the previous within-record error class.

## 3. Decision

**Reject** the continuation. No default or manifest activation changed. The
top-record interface remains an opt-in diagnostic; parent pointer weights are
retained. Do not repeat this objective unchanged. The next targeted work should
change the within-record span/query objective or exemplars, not add more
identical updates.

Evidence: [run report](../../reports/e2_pointer_scope_train_attempt02/run_report.json),
[manifest](../../reports/e2_pointer_scope_train_attempt02/extension_manifest.json).

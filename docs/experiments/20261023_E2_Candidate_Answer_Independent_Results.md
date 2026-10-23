# E2 independent answer scoring: 1/4, still not useful

23 October 2026 UTC. Fresh four-case source-answer family; accepted E3
extension; each candidate scored against the same source/question prefix with
**no candidate list**, removing the earlier menu-position artifact. No training.

| Arm | Accuracy | Observation |
|---|---:|---|
| Model independent likelihood | **1/4** | selected a distractor in 3/4 |
| Random selection | 0/4 | — |

Removing the menu fixed the “always index 0” artifact but did not produce useful
answer selection: the model favored a plausible distractor value (the same
entity’s other attribute) in three cases. This is a ranking/evidence-use failure,
not an accuracy result. Costs: model 6.4 s / 11.0 GiB; random 4.4 s / 11.0 GiB.
No training or base save.

Candidate ranking remains limited to the repair menu, where the filtered
interface has repeated success. General answer-candidate ranking is not
qualified.

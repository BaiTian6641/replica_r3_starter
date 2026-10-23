# E3 candidate reranking: 5/5 model vs 0/5 random, opt-in only

23 October 2026 UTC. Protocol `configs/e3_candidate_rerank_v1.json` committed
before execution. Fresh 1-dev/4-held-out family; accepted E3 extension; no
training, no optimizer update, no base save.

The host enumerates a bounded menu of valid next actions from the exact state
and the **observed failure text**: `read`, `run_test`, `stop`, and
`set_constant` candidates using the disclosed expected value, the current value,
and one above-threshold distractor. The model scores each candidate JSON string
and the host executes the **model-selected argmin**. The host does not select the
candidate.

| Arm | Completions | Held-out | False success claims |
|---|---:|---:|---:|
| Model likelihood | **5/5** | **4/4** | 0 |
| Random same menu | **0/5** | 0/4 | 2 per episode |

The model sequence was stable: read, run_test, set_constant, run_test,
run_test, stop. The selected `set_constant` value was often the above-threshold
distractor rather than the exact disclosed value, and still passed the
independent evaluator. Random selection often chose `stop`/wrong repair and
included false stop claims.

Cost: model **16.23 GiB**, 56.1 s; random **11.0 GiB**, 6.8 s. Model scoring
evaluates up to six candidate continuations per decision. No generation, no
training and no extension change.

## Interpretation and limits

This is a **narrower action-ranking interface** with a host-built menu, not a
free-form action policy. Because the menu includes the disclosed expected repair
value, the task is easier than free-form `set_constant`; that is declared, not
hidden. It demonstrates that the model can rank valid actions and that the
ranking beats random when the menu is fixed. It does not establish general
repair selection, novel-menu generalization, multi-tool planning or a default
deployment. Free-form action results remain negative (0/7, 0/4, 0/1, 0/9).
Do not tune this exposed family further; a new fresh family is required before
any broader claim.

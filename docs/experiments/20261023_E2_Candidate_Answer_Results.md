# E2 answer candidate ranking: position artifact, 0/4 when shuffled

23 October 2026 UTC. Fresh four-case source-grounded answer-selection data;
accepted E3 extension; model scores four candidate answers; no training.

| Arm | Accuracy | Observation |
|---|---:|---|
| Canonical order | **4/4** | model always chose index 0 |
| Random same menu | **0/4** | chose distractors |
| Model, shuffled order | **0/4** | model still chose index 0 |

The shuffled control shows the canonical **4/4 was a position artifact**: the
model assigned the lowest loss to the first candidate regardless of content
(it chose `unknown` or an owner value after shuffling). Candidate scoring did
not generalize to this answer-selection task.

Costs: canonical 7.1 s / 10.09 GiB; shuffled 6.8 s / 11.0 GiB; random 4.5 s /
11.0 GiB. No training, optimizer update or base save.

This negative is important for scope: the repair-family candidate ranking
selected multiple indices and beat random, but the same scoring mechanism is
position-sensitive on answer candidates. Do not claim general candidate-ranking
quality, and do not promote the answer-selection arm.

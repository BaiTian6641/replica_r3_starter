# E3 repair candidate order control: canonical 3/3, shuffled 3/3

23 October 2026 UTC. Fresh three-case repair family; accepted E3 extension; no
training. The same host menu was scored in canonical order and with a seed-20261023
shuffle. The correct/viable repair position changed between arms.

| Arm | Completions | Held-out | Selected repair indices |
|---|---:|---:|---|
| Canonical menu | **3/3** | 3/3 | 4, 4, 4 |
| Shuffled menu | **3/3** | 3/3 | 1, 1, 1 |

No false success claims. Both arms peaked at **15.15 GiB** and took about 31.6 s.
No training, optimizer update or base save.

This rules out a simple fixed-position artifact for the repair interface. Across
four fresh repair families the model is **17/17 (15/15 held-out)**; same-menu
random was 0/11 on the two families where it was run. The interface remains
narrow (host-built menu with disclosed repair values), opt-in, and not a
free-form action policy. Source-answer candidate ranking remains negative.

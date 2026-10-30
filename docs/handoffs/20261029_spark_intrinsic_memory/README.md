# External-agent takeover pack

Prepared on 29 October 2026 against repository HEAD
`9e75415e558c3dcff6fc85516ced160ce21a54fd`.

This pack answers the user's request for detailed prompts for external agents.
It does not launch or authorize an additional process, change an experiment,
or claim that the unfinished v12 implementation works. The project goal was
paused when this pack was prepared. No old native process needs continuation.

## How to use the prompts

For one external agent, paste the entire [coordinator prompt](00_COORDINATOR_PROMPT.md).
It is the primary continuation instruction and includes the reading order,
implementation sequence, evidence gates and reporting requirements.

For a team, use one coordinator and assign the specialist prompts below.
Each specialist must read [the shared state and boundaries](STATE_AND_BOUNDARIES.md)
before working. Implementation and independent review should be separate roles.
The coordinator alone integrates shared documentation, bindings and launch
decisions. Only the native operator may hold the native-worker lease.

| Prompt | Assignment | When to start |
|---|---|---|
| [00 — Coordinator](00_COORDINATOR_PROMPT.md) | Own the complete continuation, integration and evidence | First |
| [01 — v12 producer and CPU implementation](01_V12_IMPLEMENTATION_PROMPT.md) | Complete the frozen-source-feature candidate and real store/reader integration | Now, CPU only |
| [02 — Independent verification](02_INDEPENDENT_REVIEW_PROMPT.md) | Audit the candidate, contamination boundaries and portable verification | Review now; final checks after candidate integration |
| [03 — Native protocol and execution](03_NATIVE_EXPERIMENT_PROMPT.md) | Prepare a new exact BF16 experiment, then run it only through its gates | Prepare now; launch after CPU/review readiness |
| [04 — All-layer and cost controls](04_DENSITY_AND_COST_PROMPT.md) | Complete 2/9/36 instrumentation and shared-reader controls | CPU preparation now; native comparison after useful recall |
| [05 — Tiered memory, formation and Replica](05_REPLICA_SUCCESSORS_PROMPT.md) | Implement isolated CPU formation/validity contracts and prepare later learning | CPU preparation now; behavioral successors after useful recall |

These are work assignments, not six independent permissions to start native
jobs. CPU work may proceed in parallel only with explicit file ownership and
separate output directories. Do not allow multiple agents to edit the same
protocol, evidence index or active handoff simultaneously.

## Transfer the actual state

The committed baseline is `9e75415e`. Five v12 source files are **untracked WIP**
at this snapshot. They are listed with hashes in [wip_manifest.json](wip_manifest.json).
A Git-only clone of this HEAD will not contain them. The accompanying takeover
archive contains this prompt folder and those five exact files, with repository
relative paths. It contains no pretrained weights, environments, private source
records, historical stores, optimizer checkpoints or Git metadata.

On a different machine, obtain the repository separately, compare the current
HEAD and working tree, inspect the archive inventory, and restore its files only
where they are missing or their contents have been reconciled. Never overwrite
newer work blindly. A source-free CPU checkout is sufficient for the first two
roles. Native execution requires the qualified local source/environment or a
newly qualified environment; the archive is not a model distribution.

The WIP received syntax parsing only. Importability, functionality, gradients,
recovery, tests and native execution remain unverified. No v12 protocol, runner,
launch packet or authorization receipt exists at this snapshot. Treat apparent
implementation completeness as a review question.

The existing active repository documents and historical results were left
unchanged by preparation of this pack. Future agents should update current
navigation after implementing and measuring new work, while preserving sealed
execution bytes and historical receipts.

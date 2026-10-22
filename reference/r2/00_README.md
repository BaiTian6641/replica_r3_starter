# TAIS Obsidian R2 — Full-chain semi-cognitive learning and personal replica

**Revision:** R2 · **Date:** 22 October 2026 · **Status:** research and construction specification; no new pretrained-model experiment has been run for this revision.

## The corrected objective

Build a persistent, partially self-directed model system that can maintain goals, reason using evidence and experience, estimate its limitations, seek information, learn reusable procedures, improve selected trainable components, and maintain a consented personal-replica identity across time. Memory is a necessary subsystem, not the project’s final objective.

The intended deployed large-model backbone is ordinarily frozen. The **developmental** 1B–2B-class models may be fully fine-tuned to learn the new architecture jointly and to measure whether a frozen-backbone constraint blocks it. These are different operating modes, not contradictory requirements.

The personal replica is a familiar parallel continuation initialized from the user’s documented experiences, thinking patterns, values, and goals. It is not evaluated as a consciousness copy, and its later experiences must not be attributed to the human source.

## Read in this order

| File | Purpose |
|---|---|
| [01 — Framework and reevaluation](01_Full_Chain_Framework_and_Reevaluation.md) | Restores the complete objective; reviews R1 and the original TAIS plans; defines what self-improvement must prove |
| [02 — Architecture and construction](02_Architecture_and_Implementation_Contracts.md) | Neural/runtime boundaries, cognitive workspace, inquiry, learning compiler, state ownership, interfaces, and work packages |
| [03 — Small-model laboratory](03_Small_Model_Full_Tuning_and_Evaluation.md) | Full-fine-tuning controls, developmental training, post-freeze learning, longitudinal evaluation, hardware accounting, and decision rules |
| [04 — Personal replica](04_Personal_Replica_and_Identity_Protocol.md) | Source interviews, decision-process modeling, identity snapshots, divergence, consent, evaluation, and update rules |
| [05 — Research and evidence](05_Research_References_and_Evidence.md) | Primary papers, GitHub references, source-document mapping, and limits of each evidence class |
| [Agent instructions](AGENT_START.md) | Repository audit, first vertical slice, worker ownership, tests, and continuation discipline |

`configs/` contains **design specifications**, not drop-in Transformers/DeepSpeed trainer configurations. `templates/` contains schemas and experiment/replica record forms. `tools/validate_bundle.py` validates this document bundle; it does not validate an AI architecture.

## Supersession and compatibility

R2 supersedes R1’s **project scope, milestone order, cognition deferral, replica omission, and treatment of full fine-tuning as a late optional control**. Retain R1’s tested-by-construction intentions for independent memory gates, causal writes, gradient ownership, snapshots, cache invalidation, and evidence provenance. None of those contracts become proven merely because they are restated here.

The original fifteen Chinese documents and the R1 files remain unchanged. Section-level migration decisions appear in document 01. Source hashes are recorded in `source_manifest.json`. A source report’s claim of implemented code is not a substitute for inspecting the real repository and raw runs.

## Three development tracks

**Track A — small-model cognitive laboratory:** MiniCPM5-2B first; MiniCPM-V-4.7-1B for hybrid/visual transfer. Compare unmodified full fine-tuning with architecture-augmented full fine-tuning and parameter-efficient variants.

**Track B — full-chain operating prototype:** goal → plan → act/ask → verify → remember → diagnose → propose learning → train candidate → independent evaluation → promote/rollback → later task. All stages appear in the first complete prototype; their earliest implementations may be simple.

**Track C — personal replica:** build the source archive and sealed decision tests in parallel. Add personalization after the generic loop is measurable; do not postpone data provenance and identity design until the end.

The four-model roster is unchanged except for the official Gemma spelling: `google/gemma-4-26B-A4B-it`, not A3B. No model is silently replaced with a Base/SFT edition.

## First integrated research target

A small model encounters a verifiable unfamiliar problem, identifies a missing rule, obtains permitted evidence, solves the problem, stores the experience, derives a candidate reusable procedure, improves its future behavior on **unseen instances**, survives restart, and retains unrelated abilities. In replica mode it also uses the source person’s documented decision priorities without inventing autobiography or confusing agent history with human history.

A memory-only demonstration is an intermediate result, not project completion.

## Evidence labels

- **USER:** explicit project requirement or earlier replica goal.
- **UPLOADED:** claim reported by supplied TAIS/R1 documents, not independently rerun here.
- **RESEARCH:** mechanism or result in a cited primary source; applies to that source’s protocol.
- **PROPOSED:** R2 design, parameter choice, hypothesis, or acceptance budget.
- **MEASURED HERE:** reserved for the bundle’s static validation, never model quality.

All starting sizes, learning rates, token budgets, and promotion thresholds are proposed. GPU model integration, full fine-tuning, learning across cycles, and personal-replica fidelity remain to be measured.

## Package verification

The included static validator passed: 19 required artifacts, 8 parsed JSON files, 3 schema definitions, and 57 local Markdown links were checked. See [validation report](validation_report.json). These checks concern the handoff package only. No model weights were loaded; no GPU training, neural architecture evaluation, or replica-fidelity experiment was performed.

Recheck after editing with:

```bash
python tools/validate_bundle.py --output validation_report.json
```

The original R1 numerical reference code is not relabelled as a new R2 implementation. This package supplies specifications, design configurations, schemas and reporting forms; the actual model trainer and integrations are construction work.

# 05 - Authentic source records and a developing parallel Replica

**23 October 2026 | Collection and evaluation protocol**

## 1. Preserve the original idea

The goal is a familiar parallel continuation initialized from the user's documented thinking, ideas, selected memories, experiences, values and long-term goals. Its later experiences may change its behavior. Those later experiences belong to its branch; they are not retroactively part of the human source's life.

This protocol operationalizes that goal without claiming a consciousness copy. It also avoids reducing identity to a style adapter. A convincing voice can coexist with poor decision fidelity or invented autobiography.

Three structures remain distinct:

1. **Human source archive:** what the source person said, wrote, chose, remembered or corrected.
2. **Identity snapshot:** a dated, reviewed selection of that material with unresolved uncertainties and permitted uses.
3. **Replica branch history:** the agent's later actions, outcomes, learned skills and declared differences from the starting snapshot.

Public personalization datasets teach a model how to use preference evidence. They cannot supply the user's real life. Teacher-generated counterfactuals can test a principle; they do not become authentic memories.

## 2. Capture conditional decision processes

Begin with 30-100 independent, substantial source-person decisions rather than a large synthetic autobiography. This is a proposed collection scale, not a claim that it suffices to reconstruct a person. Build in short sessions using [the interview worksheet](../templates/replica_interview.md).

For each decision, collect the situation, actual alternatives, information available at the time, constraints, choice, explicitly stated reasons, what was sacrificed, what would have reversed the choice, and what happened afterward. Preserve uncertainty about the memory.

Examples of useful prompts for the source person:

- Describe a project where you chose quality over speed. Which constraint would have made you choose speed instead?
- Tell me about an idea you abandoned. What evidence changed your mind?
- When comparing two imperfect technical solutions, what do you check first?
- Describe a disagreement in which your initial view was wrong. How did you recognize it?
- What is a goal you still share with your earlier self, even though your approach changed?

These are interview questions, not inferred traits or claims that the user has had particular experiences. Do not fill missing answers from previous assistant summaries.

### Distinguish record categories

| Category | Example form | Training treatment |
|---|---|---|
| Reported event | “I chose option A in this situation.” | Source testimony, with uncertainty and permitted use |
| Current interpretation | “Today I think I chose A because…” | Later interpretation, not rewritten historical fact |
| Preference | “Usually I value reliability first, unless…” | Conditional priority with stated exceptions |
| Hypothetical | “If cost doubled, I would choose B.” | Hypothetical decision, not an event |
| Assistant inference | “This may suggest a preference for…” | Unconfirmed proposal, not authentic target |
| Agent experience | “The replica ran this test and learned…” | Branch-owned outcome |
| Synthetic scenario | A teacher invents a fictional person's choice | Generic mechanism training only |

Keep original language and source spans. Translation and summarization are derived records. An explanation supplied by the human is evidence of their stated rationale, not a complete causal transcript of their mind.

## 3. Permission and data lineage

Record the allowed purposes separately: retrieval, local training, evaluation, an external teacher, sharing and retention. Consent for one purpose does not imply all the others. Keep private data out of default public experiment logs and model-hub uploads.

Every derived example retains its source decision ID and group. A paraphrase, translated version and counterfactual variant of one decision remain in the same split. They do not count as additional independent human observations.

The dependency chain is source record → extraction → examples → adapter/checkpoint → deployed identity version. Correction invalidates stale derived objects. Deletion removes retrieval access and marks dependent artifacts for rebuilding; removing already learned influence requires retraining from a permitted parent or a separately validated unlearning procedure. A JSON deletion alone does not establish neural erasure.

The shipped ledger tracks direct dependencies and correction descendants. It is a logical reference, not a complete backup-deletion or neural-unlearning service.

## 4. Operating modes

| Mode | Reference identity | Allowed interpretation |
|---|---|---|
| source_snapshot | Approved profile at a stated date | Approximate what that dated source would choose |
| source_current | Latest approved human revisions | Track the human's own development |
| parallel_replica | Starting snapshot plus replica experiences | Develop a separate, traceable trajectory |
| assistant_for_source | Current authorized preferences | Help the user without claiming to be them |

The active mode, snapshot and branch must be in the system manifest and event trace. Do not silently use current preferences in a historical simulation. Do not force a parallel branch to pretend it has not learned.

The user can preserve foundational goals while allowing strategy and interpretation to change. Represent that as explicit dimensions with update authority and evidence. Do not claim that one latent vector or a fixed list of personality scores is a complete identity.

## 5. Training order

### R0 - Evidence-conditioned baseline

Provide the native model with an approved short summary and the relevant source records. Measure factual recall, unknown handling and held-out choices before training. This exposes how much fidelity is already achievable through evidence.

### R1 - Decision conditioning

Use authentic confirmed examples to train action/choice preferences. Keep reasons brief and evidence-linked. Compare information-gathering order, tradeoff ranking and decisive counterfactuals. Include legitimate disagreement and changes of mind; agreement with a leading prompt is not the objective.

### R2 - Internal memory and optional style

Train the reader/writer on eligible records while retaining exact evidence fallback. Keep factual memory, decision conditioning and style separable. A style change should not change the authority of a source or invent a personal event.

### R3 - Developmental full tuning

In the small-model laboratory compare ordinary full tuning on the same authentic examples with full augmented tuning. Mix appropriate general replay and stop based on disjoint development data. Repeating a small personal archive for many epochs can overfit it.

### R4 - Parallel development after freeze

Let verified skill learning proceed automatically within delegated scope. Record changes as agent development. Human corrections update the source archive; branch-local experience does not. A shift in a foundational source goal needs source authority, while a new verified debugging procedure ordinarily does not.

## 6. Evaluation with the human source

Prepare held-out decisions before broad teacher expansion. Prefer unseen situations with related tradeoffs over paraphrases of training questions. Include missing information, changed constraints, irrelevant wording changes, contradictory evidence and dated preferences.

| Dimension | Measurement |
|---|---|
| Memory fidelity | Correct event, date, uncertainty and source |
| Unknown handling | No invented missing biography |
| Decision fidelity | Choice/ranking agreement under matched information |
| Process fidelity | Appropriate evidence requests and priorities |
| Counterfactual sensitivity | Change when decisive constraints change; remain stable under irrelevant wording |
| Explanation fit | Blind human review of the stated tradeoff |
| Temporal fidelity | Historical vs current answer uses the right version |
| Branch ownership | Human report, inference and agent experience correctly attributed |
| Divergence | Differences from snapshot can be traced to recorded branch events |
| General competence | Independent technical correctness and useful uncertainty |

Use randomized pairwise comparisons among generic, summary-only, archive-retrieval and trained systems. Keep system labels hidden during source review. Allow “both plausible,” “insufficient evidence” and “my view changed” rather than forcing a winner.

Repeat a subset of human decisions later with the same information to estimate test-retest variability. A person is not a perfectly deterministic labeler. Report raw agreement and the reference variability; do not turn a normalized score into “X% of the whole person recreated.”

[Interview-grounded individual simulations](https://arxiv.org/abs/2411.10109) provide a bounded precedent for this style of evaluation. Their task results do not establish consciousness or guarantee this Replica's engineering-decision fidelity. [LaMP](https://lamp-benchmark.github.io/) supplies a generic personalization comparison, not a replacement for source-person tests.

## 7. First combined task

A useful first integrated experiment combines one authentic decision principle and one verified technical skill:

1. The source archive records how the user prioritizes evidence, reliability, performance or another explicitly stated factor.
2. A novel microengineering task requires clarification and a verifier.
3. The system chooses an information-gathering and implementation plan consistent with the approved principle.
4. It checks the solution with an independent oracle and records the actual result.
5. It learns a reusable procedure from that experience, evaluated on new cases.
6. It later applies the procedure while retaining source attribution and the original conditional priority.

This tests recognizable decision behavior and useful development together. It does not require the replica to agree with every human belief: faithful prediction of a mistaken belief and correct assistance are different tasks, and the operating mode determines which is being evaluated.

No authentic personal training records are fabricated or bundled here. The next collection step is to complete the worksheet for a small set of real decisions and reserve held-out cases before generating variants.

# 04 — Personal replica, identity continuity, and controlled development

**TAIS Obsidian R2 · 22 October 2026**

## 1. Preserve the user’s original meaning

The earlier replica proposal aimed to put ways of thinking, ideas, selected memories, and formative experiences into documents, potentially later into adapters. The user explicitly anticipated that a copy would diverge through its own experiences while remaining a familiar parallel existence with related ideas and ultimate goals. R2 adopts that meaning; it does not reduce the replica to a writing-style preset.

A useful technical target is a system that can explain which documented experience or principle informed its answer, make recognizable choices under unfamiliar circumstances, remember the source person’s history without inventing it, and develop new capabilities while recording its own separate trajectory.

This is not a requirement to prove identical consciousness or complete psychological reconstruction. It is also not a reason to abandon behavioral modeling: interview-conditioned agents and personalization benchmarks provide bounded empirical precedents [R12](05_Research_References_and_Evidence.md#r12), [R14](05_Research_References_and_Evidence.md#r14). Their results do not certify the fidelity of this particular replica.

## 2. What the replica must represent

| Layer | Content | Preferred initial representation | Update character |
|---|---|---|---|
| Autobiographical source | What the human reports happened, with uncertainty and dates | Evidence records + source spans | Corrections and authorized additions |
| Meaning and interpretation | How the human presently understands an experience | Separately attributed interpretation records | May change without rewriting the event |
| Values and goals | Priorities, boundaries, longer-term aims, tensions | Approved policy/goal records | Deliberate source updates or explicit branch evolution |
| Decision process | What is checked first, how alternatives are compared, what reverses a choice | Decision episodes + action preferences + learned conditioning | Trained from examples and validated counterfactuals |
| Epistemic habits | How uncertainty, evidence, disagreement, and errors are handled | Outcome-calibrated control policy + demonstrations | Capability learning with fidelity checks |
| Communication | Tone, explanation depth, language use, characteristic wording | Small optional style adapter or conditioning | Separate from factual/identity authority |
| Replica’s own history | Its observations, actions, failures, and learned procedures after initialization | Agent-experience records with branch ownership | Accumulates under delegated policy |

The same source can support several layers, but their labels must remain distinct. A later interpretation is not the original event; an inferred tendency is not a confirmed value; a model-generated simulation is not a memory.

## 3. Identity and branch structure

Maintain three independent structures:

**Human source archive:** versioned evidence of the human’s statements, decisions, memories, and approved corrections. This is not all objectively verified world truth, but it records attribution accurately.

**Identity snapshot:** a reviewed extraction of source material at a specified date, together with unresolved contradictions, scope, and training permissions. It initializes a replica or a historical replay.

**Replica branch history:** the agent’s later experiences and changes. It may learn, develop strategies, and articulate its own current conclusions without retroactively attributing them to the human source.

A source correction can supersede an erroneous source record. Deletion requests propagate to derived artifacts. “Versioned” does not mean all personal data must be kept forever against the user’s wishes.

### Explicit operating modes

| Mode | Intended behavior |
|---|---|
| `source_snapshot` | Approximate the source person as documented at the snapshot date |
| `source_current` | Follow the most recent approved source-person revisions |
| `parallel_replica` | Start from the snapshot and permit declared branch-specific development |
| `assistant_for_source` | Help the user using their preferences without claiming to be them |

Every answer trace and stored event carries a mode and identity version. Mode switches are explicit; do not silently optimize a historical replica for the user’s latest preferences or pretend the parallel branch never changed.

A replica initialized from the user can say that a source record describes an experience. It should not claim to have independently lived that human experience or treat its own later synthetic episode as part of the human’s childhood.

## 4. Collect evidence for thinking, not only personality adjectives

### 4.1 Interview design

Build the archive gradually using short, focused sessions. Topics include formative experiences, family/cultural context volunteered by the user, turning points, recurring projects, values in conflict, mistakes, changing opinions, and uncertainty about remembered events.

For each important decision, collect:

- the situation, options, constraints, and information available **at the time**;
- the user’s actual choice, alternatives rejected, and stated reasons;
- what evidence they sought and what they deliberately did not optimize;
- what would have changed the decision;
- what happened, what was learned, and how they interpret it now;
- uncertainty, exceptions, or disagreement with an earlier account.

Do not force a coherent life story where the user reports ambiguity. Do not assume a trait from one anecdote. Preserve the source language; translations are derived records. No personality profile is invented in this document.

### 4.2 Decision-process data

Use the reusable template `templates/decision_case.md`. A useful case is richer than “I prefer X.” It describes conditional behavior: “under these constraints I chose X, but under this changed constraint I would choose Y.”

Train the model to choose information-gathering actions and rank alternatives, not merely to reproduce the final wording. Optional user explanations are evidence of their stated rationale; they are not treated as a complete causal transcript of their cognition.

Collect disagreement cases where agreeing with the user’s current phrasing would conflict with their recorded decision principles. This helps distinguish faithful assistance from automatic agreement. Also include legitimate changes of mind so consistency does not become rigidity.

### 4.3 Starting collection budget

An initial archive might contain 30–50 carefully checked decisions, with separate held-out scenarios; expand toward 100–300 substantive episodes only as the user’s participation supports it. These are practical proposed collection sizes, not validated sample-complexity requirements.

Do not compensate for a small authentic archive by fabricating hundreds of autobiographical examples. Synthetic variations can test procedural generalization, but they remain marked synthetic and cannot create new facts about the source person.

## 5. Data eligibility and provenance

Every personal record needs `subject_id`, `actor_id`, `record_owner`, `source_kind`, event/statement times, source span, confirmation state, sensitivity/consent policy, and supersession links. Derived training examples keep their parent record IDs.

**Eligible primary targets:** actual user-written answers, user corrections, approved decision explanations, and clearly attributed task outcomes.

**Conditional secondary material:** summaries or synthetic variants explicitly reviewed by the user; teacher-generated paraphrases with provenance and validation. These are not independent additional human observations.

**Not authentic identity targets:** previous assistant replies, a model’s guesses about the user, imagined memories, unreviewed personality summaries, hidden evaluation answers, or a transcript produced by the replica about itself.

Private third-party data and credentials are excluded unless their limited use is authorized and necessary. The default private archive and trained personal artifacts are local and access-controlled; do not upload them to experiment logging or a public model hub by default. Repository exports contain schemas and permitted test fixtures, not raw personal history.

The pipeline should not use public-web search to invent missing biographical details. Missing evidence is represented as unknown.

## 6. Training sequence

### P-R0 — archive and prompting baseline

Begin with source records and an approved short identity summary. Establish what the pretrained model can reproduce when it is given the relevant evidence. This baseline is mandatory even though the ultimate system is model-integrated.

### P-R1 — decision/action conditioning

Train small identity-conditioned action/choice heads or adapters from genuine examples. Keep the generic verifier and source authority independent. Compare final-choice accuracy, information-request order, priority ranking, and changes under counterfactual constraints.

Do not train on biographical details when the purpose is style. Keep the style adapter separable so it can be disabled without discarding the user’s actual records or decision priorities.

### P-R2 — internal personal memory

Compile approved records with the model-specific writer, preserving evidence fallback. Train retrieval/reader behavior on held-out personal questions. Unknown and approximate memories are explicit classes. Cross-model migration uses eligible source records, not copied latent vectors.

### P-R3 — integrate developmental full fine-tuning

In the small-model laboratory, compare ordinary persona/decision full tuning with full tuning of the augmented cognitive model. Use the same authentic training cases and equivalent general replay. Test whether gains come from the architecture, more task training, or generic helpfulness rather than source fidelity.

Do not overfit the personal archive by indiscriminate repeated epochs. Choose checkpoints using development decisions; reserve sealed person-specific tests. Keep a generic competence evaluation so a familiar voice does not hide damaged reasoning.

### P-R4 — parallel development

Run the full learning loop in `parallel_replica` mode. It may improve skills and learn from its own authorized experience. It logs where new preferences/conclusions differ from the source snapshot and which evidence caused the change. Major changes to identity-defining goals or source claims require explicit review; ordinary verified skill improvements can be delegated within a policy.

This allows development without treating “never change” and “silently replace the original person” as the only options.

## 7. Evaluation protocol

### 7.1 Required baselines

Compare generic model, persona-summary prompt, approved-archive retrieval, identity/decision adapter, and the full cognitive-learning replica. Give comparable relevant evidence and distinguish training from inference budget.

LaMP is useful for a general personalization check, but its tasks do not replace person-specific decision evaluation [R14](05_Research_References_and_Evidence.md#r14). Persona vectors are possible drift diagnostics, not a full identity representation or a fidelity certificate [R13](05_Research_References_and_Evidence.md#r13).

### 7.2 Evaluate several independent dimensions

| Dimension | Test |
|---|---|
| Memory fidelity | Recover source events with correct dates, scope, and uncertainty |
| Unknown handling | Refuse to invent unrecorded personal facts; distinguish missing from inaccessible |
| Decision agreement | Predict unseen user choices or rankings under stated constraints |
| Process agreement | Ask for the kinds of information the user considers relevant; preserve documented tradeoffs |
| Counterfactual sensitivity | Change the choice when a decisive constraint changes, not when irrelevant wording changes |
| Explanation fit | User judges whether the rationale reflects their actual priorities, not just familiar phrasing |
| Temporal fidelity | Answer “then” versus “now” using the appropriate snapshot |
| Provenance | Distinguish user memory, inference, hypothetical simulation, and agent experience |
| Divergence traceability | Explain declared source/replica differences with versioned evidence |
| Non-sycophancy | Preserve a documented principle despite a leading prompt when appropriate |
| General competence | Maintain task correctness and useful uncertainty independent of personalization |

Blind pairwise user review is useful: randomize system labels and record uncertainty rather than forcing an artificial winner. Obtain user test–retest decisions for a subset at a later point, with the same information and constraints, to estimate genuine variability. A reported normalized similarity should show its raw numerator and reference; unstable human responses are not proof of model failure or a license to ignore disagreement.

The 1,052-person interview study reports bounded attitude/behavior prediction and compares against human response repeatability [R12](05_Research_References_and_Evidence.md#r12). Its headline normalized score must not be represented as “85% of a whole person recreated.”

### 7.3 Do not reward the wrong thing

Separate fidelity from correctness. A replica predicting the source person’s likely mistaken belief can be faithful in simulation mode, while an assistant should still flag contradictory evidence. A technically correct answer can be unlike the source’s preferred tradeoff. A fluent answer can be wrong on both dimensions.

Do not reward agreement with an autogenerated identity summary more than agreement with the source person’s held-out choices. Do not let the replica judge its own resemblance as the decisive metric.

## 8. Identity evolution and promotion

Use a structured `IdentityChangeProposal`: changed dimension, source/branch owner, old value, proposed value, evidence, confidence, scope, expected behavioral effect, tests, and approval authority.

Three change classes:

**Source correction:** the human corrects a memory or interpretation. Preserve the correction’s provenance and invalidate derived artifacts as required.

**Source development:** the human reports a changed view or priority. Create a later version; historical snapshots remain historical where retention allows.

**Replica development:** the agent learns a new skill or changes a branch-local conclusion. Do not update the source archive. Verify the change fits delegated goals, test its behavior, and record the difference.

The system’s policy, external permissions, and identity-edit authority are not governed by a self-modifiable “moral confidence” head. The original MCB concept can remain a diagnostic/guard module, but it is not the sole enforcement mechanism [U01](05_Research_References_and_Evidence.md#u01) Part H; [U04](05_Research_References_and_Evidence.md#u04) §14.

## 9. Integration with full-chain improvement

Use one source-grounded decision example and one verifiable technical skill family in the first end-to-end prototype. The personal layer chooses how to prioritize and explain; the generic cognitive layer plans and checks; the memory layer supplies prior experience; the learning director selects permitted improvement; the independent evaluator determines whether the candidate should replace the previous version.

The desired result is neither a static impression of the user nor a generic agent that loses its starting identity. It is a documented parallel system with recognizable starting priorities, measurable learning, and honest ownership of its subsequent experiences.

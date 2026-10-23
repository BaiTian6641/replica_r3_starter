# Replica architecture research and design

**Reviewed 22 October 2026, America/Chicago / 23 October UTC.** This is the
current detailed design for rebuilding the research starter into the intended
model-integrated system. It follows the user's request to prioritize the real
architecture over further isolated short-answer probes.

The objective is a pretrained LLM enhanced with complex memory, a parallel
cognitive core, native memory access ports, self-evaluation, bounded learning
and an authentic personal Replica branch. MiniCPM5-2B is the first technology
validation model; MiniCPM-V-4.7-1B is the second, with its own native port.

**Implementation follow-up:** [E1/E2 implementation and qualification](07_E1_E2_Implementation.md) records the connected code and completed MPS qualification. Slot-only answers are 4/4 with source-swap/restart controls; reranking quality remains open. See the [native results](../experiments/20261023_E1_E2_Native_Results.md). The research chapters below remain the design baseline.

## Read in this order

| Document | What it specifies |
|---|---|
| [1. System blueprint](01_System_Blueprint.md) | Complete architecture, 18-component inventory, concrete dimensions, execution order, ownership and use |
| [2. Backbone integration](02_Backbone_Integration.md) | Exact text/VLM topologies, depth ports, contextual writer, gradients, positions/caches and resource arithmetic |
| [3. Memory and runtime compilation](03_Memory_and_Runtime_Compilation.md) | Evidence, per-record latents, retrieval, fast memory, persistent images, lookup, skills, compilation and invalidation |
| [4. Semi-cognition, learning and Replica](04_Semi_Cognition_Learning_and_Replica.md) | TAIS inheritance, task/core/action loop, capability, real candidate learning and personal identity integration |
| [5. Implementation sequence](05_Implementation_Sequence.md) | Six connected build packages, code migration, training/evaluation and hardware plan |
| [6. Research decisions](06_Research_Decisions.md) | Primary sources, distinctions, adaptation choices and unresolved questions |

Each detailed chapter contains block diagrams alongside component inputs,
outputs, state ownership, training/update cadence and use. The
[rendered diagram catalog](../../diagrams/architecture_v1/README.md) provides
standalone SVG and editable Mermaid versions. The
[machine-readable design](../../configs/architecture_design_v1.json) links every
component to its design section and diagram.

## Main decisions

- Retain pretrained native models and train interfaces plus selected development
  co-adaptation. Do not build another language foundation from random weights.
- Use multiple native access ports and a shared parallel core, coupled to exact
  task state and real action/outcome feedback.
- Combine exact temporal records, learned record access, scoped neural memory,
  reusable procedures and separate identity/branch history.
- Make runtime compilation produce named, versioned artifacts. Acquiring a
  record, updating fast state, compiling a skill and optimizing model parameters
  are different operations with different evidence requirements.
- Start with a clear known-input sense pass, a bounded core update and an action
  pass. Preserve causal ownership before optimizing the schedule and caches.
- Build the text implementation first, then adapt the same contracts to the
  hybrid VLM. Larger-model transfer requires fresh model-specific interfaces.

The concrete text starting design is width-256 shared state, eight workspace
slots, two core iterations and ports after native blocks 13/27, with 41 as observer.
The memory bank contains at most 16 **memory units** × four slots. A memory unit
is one bounded source record/span; a long source split into several spans uses
several selection units. The initial writer uses 128-token spans. The optional
persisted-image replay path separately allows up to 256 tokens per event chunk;
these are distinct budgets, never permission to silently double a read bank.

Canonical interface names are `RequestContext` (mutable request ownership),
`DecisionFrame` (immutable decision inputs), and `InjectFrame` (its per-port
read-only view). “Control frame” in descriptive prose refers to `DecisionFrame`.
The reference sense pass disables new injections; the core receives prior
task/workspace state explicitly. The answer/action pass uses a fresh native
cache and the new frame.

## What this revision corrects

The earlier documents named **Mem-T**, a learned agent over a memory database.
The user's model-modification direction also requires **MemT-1.0**, a distinct
memory-native retrofit. Both are now researched and assigned separate roles.

The supplied [TAIS-Obisian repository](https://github.com/BaiTian6641/TAIS-Obisian/tree/85c134e857299431d1504636115d273a3de48bf2)
is now inspected at a pinned revision. It contains real source implementations
and research reports; older notes that no matching public upstream had been
identified describe the prior review, not this one. Its custom-model mechanisms
inform the design; its reported results do not qualify our MiniCPM implementation.

At the design handoff, native evidence covered the earlier last-block adapter
and preserved response-contract failures. The subsequent E1/E2 implementation
now adds measured native source-to-port integration. This remains a narrow
synthetic diagnostic, with no personal-fidelity or self-improvement result.

## Immediate implementation

**E1/E2 implementation and bounded native diagnostic are complete** (reranking
quality remains open). Their delivered scope is: explicit request/manifest interfaces, source-only
contextual record encoding and two native memory ports. Next connect the shared
core, task journal and real neural action/outcome loop in E3. Integrate temporal
memory/compilation and actual candidate learning in E4/E5. Source admission and
fair-comparison design proceed alongside construction; they do not require more
rounds of the same sixteen-case prompt experiment before architecture work.

No native jobs or checkpoint downloads are part of this research pass. Reuse
the existing source and preserve previous evidence. See the [review record](../../reports/architecture_research_20261023/review.json)
for source pins, diagram validation and preserved-code checks.

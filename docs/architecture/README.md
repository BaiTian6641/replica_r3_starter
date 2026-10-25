# Replica architecture research and design

For the requested detailed Spark target analysis, see the
[integrated Spark research report](../research/sparkx_2_6_4b_20261024/README.md):
repository audit, spatial/temporal vision, Titans/HOPE-inspired learning,
semi-cognitive integration, staged evaluation and research evidence. It includes
the current FP8/BF16 preparation and remaining recovery requirements below.

**Current priority, 25 October 2026, reviewed HEAD `7c7018e`:** remaining
caller-context verification, complete route-sequence auditing and representation-
preserving rebuild fixes, then isolated MiniCPM5 V0. Follow with V1/V2 visual
fine-tuning, qualified visual memory and bounded RSI. Spark FP8/BF16 preparation
and dataset curation proceed independently now. MiniCPM5 remains the baseline;
the 50 GiB native guard is unchanged.

The v5 writer repair completed and rejected: fast and disabled both 56/192,
zero learned factual recall. Exact context is 184/192 and all 128 facts, but
fails the reference's general/nonregression gates. Source packaging, changed-span
checks and original route counters are repaired. The review reproduced the
remaining narrower defects and independently verified 49 focused tests, source/
data pins and canonical scoring. See [recovery](17_Memory_Reliability_Recovery.md)
for scope and [execution state](../../execution_state.md) for native evidence.

The complete system plan is in
[document 19](19_Integrated_Memory_Learning_and_RSI_Design.md), with
[document 20](20_Spark_X2_6_FP8_Preparation_Plan.md) for the requested official
Spark FP8 base and independent BF16 control, and
[the dataset curriculum](../08_Dataset_Acquisition_and_Curriculum.md) for 34
catalogued sources, 22 pinned metadata records and staged acquisition. No weights
or dataset assets were acquired or newly admitted by this documentation update.

MiniCPM5 receives its own encoder/projector interface. MiniCPM-V remains a
separate optional pretrained reference/teacher. Spark needs a new width-2560
connector and dedicated native adapter; its preparation does not wait for RSI,
and its native/backend gates do not delay MiniCPM5.

## Read in this order

| Document | What it specifies |
|---|---|
| [1. System blueprint](01_System_Blueprint.md) | Complete architecture, 18-component inventory, concrete dimensions, execution order, ownership and use |
| [2. Backbone integration](02_Backbone_Integration.md) | Exact text/VLM topologies, depth ports, contextual writer, gradients, positions/caches and resource arithmetic |
| [3. Memory and runtime compilation](03_Memory_and_Runtime_Compilation.md) | Evidence, per-record latents, retrieval, fast memory, persistent images, lookup, skills, compilation and invalidation |
| [4. Semi-cognition, learning and Replica](04_Semi_Cognition_Learning_and_Replica.md) | TAIS inheritance, task/core/action loop, capability, real candidate learning and personal identity integration |
| [5. Implementation sequence](05_Implementation_Sequence.md) | Six connected build packages, code migration, training/evaluation and hardware plan |
| [6. Research decisions](06_Research_Decisions.md) | Primary sources, distinctions, adaptation choices and unresolved questions |
| [Research cross-check and backbone plan](15_Research_Crosscheck_and_Backbone_Plan.md) | Research versus implementation; baseline, official Spark FP8/BF16, memory, vision and bounded RSI |
| [Loop-agent execution instructions](16_Loop_Agent_Instructions.md) | Ready-to-use bounded implementation loop, first package, evidence gates, preservation rules and reporting |
| [Reproducibility and memory recovery](17_Memory_Reliability_Recovery.md) | Caller-bound contexts, exact schedule, representation rebuild and independent memory diagnosis |
| [MiniCPM5 vision and continual learning](18_MiniCPM5_Vision_Training_and_Continual_Learning.md) | V0–V4 visual input, alignment/instructions, visual memory and verified RSI |
| [Integrated memory, learning and RSI design](19_Integrated_Memory_Learning_and_RSI_Design.md) | Ownership, dynamic routing, compilation, learning and qualification roadmap |
| [Spark FP8/BF16 preparation](20_Spark_X2_6_FP8_Preparation_Plan.md) | Official pins, precision/backend gates, new vision/memory interfaces |
| [Dataset acquisition and curriculum](../08_Dataset_Acquisition_and_Curriculum.md) | 34 sources, intake states, staged quantities, licenses, dedup and evaluators |

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
- Repair the text memory/evidence foundation, then train MiniCPM5's new visual
  interface. Visual memory and RSI require independent quality gates. Optional
  pretrained VLM and later Spark ports need model-specific interfaces.

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

Follow [document 17](17_Memory_Reliability_Recovery.md) for R1 caller-bound context,
R2 complete schedule audit and R3 representation-preserving rebuild. Then follow
[document 18](18_MiniCPM5_Vision_Training_and_Continual_Learning.md) for isolated
V0. The rejected v5 package is complete; another blind memory-training arm is
not a V0 prerequisite. Useful memory remains a gate for learned V3 and integrated
RSI. Continue diagnostic memory probes, data intake and Spark source/backend
preparation independently. The [loop prompt](16_Loop_Agent_Instructions.md)
controls the next concrete work, while component chapters retain their detailed
contracts and historical scientific dependencies.

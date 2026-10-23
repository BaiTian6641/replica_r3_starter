# 07 - Architecture diagrams

**Current component diagrams:** [architecture v1 catalog](../diagrams/architecture_v1/README.md)
contains rendered SVGs and editable Mermaid for the full
[research design](architecture/README.md). It covers the two backbones, memory
ports, persistent stores, shared core, action/learning loops and Replica.
The original eight-page atlas below remains preserved as the earlier framework.

**All diagrams describe proposed research architecture. Shipped CPU contract tests do not establish model-quality improvement.**

[Download the eight-page vector atlas](../diagrams/Replica_R3_Architecture_Atlas.pdf). Every diagram is also available as SVG, PNG and editable Mermaid source. The PDF/SVG layouts show implementation-status colors; Mermaid prioritizes relationships.

## Complete system

[SVG](../diagrams/01_full_system.svg) | [PNG](../diagrams/01_full_system.png) | [Mermaid source](../diagrams/01_full_system.mmd)

```mermaid
flowchart TD
    U[Goal and available observations] --> C[Controller and task workspace]
    E[Scoped evidence records] --> R[Trained memory reader]
    I[Approved identity and branch history] --> C
    R --> B[Backbone plus trainable extensions]
    C --> B
    B --> A{Answer or act}
    A --> O[User-facing answer]
    A --> T[Permitted tools]
    T --> V[Verified outcome]
    V --> C
    V --> E
    V --> L[Learning proposal]
    L --> Q[Candidate training and independent evaluation]
    Q --> P{Accept or reject}
    P -->|accept| B
    P -->|reject| L
```

## Native decoder integration

[SVG](../diagrams/02_native_llama_integration.svg) | [PNG](../diagrams/02_native_llama_integration.png) | [Mermaid source](../diagrams/02_native_llama_integration.mmd)

```mermaid
flowchart TD
    H[Native hidden states] --> N[Complete native decoder block]
    N --> S[Add independent residual deltas]
    S --> D[Remaining native decoder and output]
    E[Past source embeddings] --> W[Trainable record writer]
    W --> R[Memory reader]
    N --> R
    R --> S
    P[Known prompt embeddings] --> C[Bounded recurrent workspace]
    C --> X[Workspace reader]
    N --> X
    X --> S
```

## Separate state and update timescales

[SVG](../diagrams/03_memory_timescales.svg) | [PNG](../diagrams/03_memory_timescales.png) | [Mermaid source](../diagrams/03_memory_timescales.mmd)

```mermaid
flowchart TD
    O[New permitted observation] --> W[Task workspace]
    O --> F[Private fast-state candidate]
    O --> E[Evidence record proposal]
    W --> A[Current action or answer]
    F --> B[Chunk-end commit barrier]
    B --> FN[Next chunk private state]
    E --> H[Host validates source and scope]
    H --> DB[Durable evidence ledger]
    DB --> L[Learning director]
    L --> C[Candidate skill or sidecar update]
    C --> V[Independent comparison]
    V --> P[Versioned extension publication]
    BASE[Frozen deployed base] --> A
```

## Causal support and shared fast-state barrier

[SVG](../diagrams/04_causal_support_query.svg) | [PNG](../diagrams/04_causal_support_query.png) | [Mermaid source](../diagrams/04_causal_support_query.mmd)

```mermaid
flowchart TD
    P[Already observed support and prompt] --> C[Decision context]
    C --> Q[Current query and causal decoding]
    S[Chunk-start state] --> L1[Depth port 1 reads]
    S --> L2[Depth port 2 reads]
    L1 --> Q
    L2 --> Q
    Q --> O[Fix current chunk outputs]
    L1 --> W1[Deferred write contribution]
    L2 --> W2[Deferred write contribution]
    O --> B[Commit barrier]
    W1 --> B
    W2 --> B
    B --> N[Next chunk state]
    Y[Held-out answers and future events] --> X[Evaluator only]
```

## Autonomous learning and promotion

[SVG](../diagrams/05_autonomous_learning_loop.svg) | [PNG](../diagrams/05_autonomous_learning_loop.png) | [Mermaid source](../diagrams/05_autonomous_learning_loop.mmd)

```mermaid
flowchart TD
    E[Actual experience and outcome] --> D[Diagnose reusable success or failure]
    D --> P[Choose bounded practice proposal]
    P --> T[Train isolated candidate]
    T --> V[Independent task and retention evaluation]
    V --> G{Release decision}
    G -->|accept| M[Publish compatible manifest]
    G -->|reject or inconclusive| D
    M --> U[Later unseen tasks and restart tests]
    U --> E
    M --> R[Retain rollback revision]
    R -->|later regression| B[Restore prior compatible manifest]
```

## Source and parallel Replica identity

[SVG](../diagrams/06_replica_identity_lineage.svg) | [PNG](../diagrams/06_replica_identity_lineage.png) | [Mermaid source](../diagrams/06_replica_identity_lineage.mmd)

```mermaid
flowchart TD
    H[Human testimony and real decisions] --> A[Source archive]
    A --> S[Reviewed dated snapshot]
    S --> P[Parallel Replica branch]
    P --> E[Agent actions and outcomes]
    E --> B[Branch-owned experience and development]
    B --> P
    H --> C[Authorized source correction]
    C --> A
    C --> I[Invalidate dependent artifacts]
    S --> T[Held-out source-fidelity tests]
    P --> T
    T --> D[Report fidelity and declared divergence]
```

## Two-stage experimental progression

[SVG](../diagrams/07_two_stage_roadmap.svg) | [PNG](../diagrams/07_two_stage_roadmap.png) | [Mermaid source](../diagrams/07_two_stage_roadmap.mmd)

```mermaid
flowchart TD
    P[Pin environment and native baseline] --> S[Small text model laboratory]
    S --> D0[D0 ordinary full tuning]
    S --> D1[D1 frozen base plus modules]
    S --> D2[D2 full augmented tuning]
    D0 --> C[Matched task and retention comparison]
    D1 --> C
    D2 --> C
    C --> F[Freeze selected developmental base]
    F --> L[Ten post-freeze learning cycles]
    L --> V[Small native VLM port]
    V --> G{Mechanism earns its cost}
    G -->|yes| M[Fresh large-model interfaces]
    G -->|no or unclear| R[Fix bottleneck or report negative result]
    M --> B[Longer large-model learning study]
```

## Source lineage before generation

[SVG](../diagrams/08_dataset_pipeline.svg) | [PNG](../diagrams/08_dataset_pipeline.png) | [Mermaid source](../diagrams/08_dataset_pipeline.mmd)

```mermaid
flowchart TD
    S[Permitted sources and deterministic worlds] --> R[Source registry and lineage groups]
    R --> T[Training groups]
    R --> D[Development and calibration groups]
    R --> A[Sealed evaluation groups]
    T --> G[Teacher expansion with source IDs]
    G --> Q[Unverified candidate examples]
    Q --> V[Independent solver or source review]
    V -->|pass| F[Versioned training export]
    V -->|fail| X[Reject or repair with evidence]
    F --> M[Candidate model training]
    D --> P[Development feedback]
    M --> P
    A --> E[Independent final audit]
    M --> E
```

## Reading the evidence correctly

The native Llama integration, memory primitives and trusted-process publication utilities have numerical/software tests. The complete learned controller, autonomous neural curriculum, VLM port and larger-model transfer remain proposed integration work. The source/prompt boundary excludes future target tokens from inputs; training may still use later target labels through the declared loss.

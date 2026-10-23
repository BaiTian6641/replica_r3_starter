# Research findings and architecture decisions

Reviewed 22 October 2026, America/Chicago / 23 October UTC. Claims below refer to
the inspected version. The proposed integration is our design synthesis, not a
claim that any paper validates the complete Replica architecture.

## 1. The original TAIS implementation is a substantive reference

The user supplied [BaiTian6641/TAIS-Obisian](https://github.com/BaiTian6641/TAIS-Obisian/tree/85c134e857299431d1504636115d273a3de48bf2).
The inspected commit is `85c134e857299431d1504636115d273a3de48bf2`, dated 31 July 2026.
A source-only temporary checkout and selected text blobs were inspected;
upstream code was not executed and no weights were downloaded.

| Upstream idea / code | Carry into this project | Adaptation decision |
|---|---|---|
| Kernel sense/route/inject and HRL | Explicit model-to-runtime bridge | Learn native MiniCPM projections and typed routes rather than replace its decoder |
| Thought core and integration | Bounded parallel state evolution that changes native representations | Shared core at two declared ports; outcome-calibrated stopping and real task feedback |
| CA3 association, pager, page table and CA1 gating | Candidate retrieval, scoped address resolution and usefulness selection | Require authoritative source/producer binding; novelty and similarity do not certify truth |
| Memory layers and persistent stream | Neural read/write paths and retained working state | Request-owned functional state, causal writes, replayable persistent images |
| Inquiry and knowledge-awareness components | Decide when to ask, retrieve or verify | Actual tool/source outcomes; calibrated capability instead of mock confidence |
| Sleep consolidation | Scheduled transformation of experience into reusable artifacts | Explicit record, procedure and neural candidate outputs; real optimization for weight-learning claims |
| Dynamic concept slots | Source-bound concept access without repeating all source text | Keep tokenizer stable initially; concept aliases and sparse lookup remain distinct from exact factual storage |

The [cognition chapter](04_Semi_Cognition_Learning_and_Replica.md) traces inspected
files and prototype boundaries. Its source manifest distinguishes directly read
files from other fetched text. A later source file can supersede an older status
note; code presence and author-reported scores remain separate evidence classes.

## 2. Which research belongs at which boundary

| Work and primary source | Role in this architecture | What remains to establish locally |
|---|---|---|
| [MemT-1.0 v2](https://arxiv.org/html/2610.26825v2), [official repository](https://github.com/MemTensor/MemT-1.0) | Native adaptive memory inside an existing backbone; separate store/use phases | Useful small-model adaptation at affordable training cost and MPS-compatible operators |
| [Mem-T v2](https://arxiv.org/abs/2601.23014v2), [official repository](https://github.com/yanweiyue/Mem-T) | Learned construction/retrieval operations over a hierarchical store | A trained operation policy linked to our evidence schema and native ports |
| [Titans](https://arxiv.org/html/2501.00663v1) and [TTT](https://arxiv.org/html/2407.04620v1) | Learnable associative state and inner/outer update separation | Useful storage/read representations, stability, interference and adaptation cost |
| [uTTT](https://arxiv.org/html/2610.05484v1) | Shared state across native depths with explicit write scheduling | Benefit over equal-capacity private states in a retrofit, with causal cached execution |
| [Engram v2](https://arxiv.org/html/2601.07372v2) | Conditional static lookup capacity for repeated token patterns | Additional benefit over record memory, plus table/optimizer/bandwidth cost |
| [Nested Learning / HOPE](https://arxiv.org/html/2512.24695v1) | Distinct update frequencies and objectives | A bounded multirate implementation that improves retention/usefulness |
| [SEAL v2](https://arxiv.org/abs/2506.10943v2) | Learned self-edit/practice proposals rewarded after real adaptation | Positive later gains, candidate isolation, rejection and learning-budget accounting |
| [Voyager](https://arxiv.org/abs/2305.16291v2) | Reusable executable skills and experience-guided practice | Transfer and compositional reuse on our real tasks; separate from neural gains |
| [Continuous Thought Machines](https://arxiv.org/abs/2505.05522), [Coconut](https://arxiv.org/abs/2412.06769) | Optional comparisons for internal iterative computation | A small recurrent workspace is neither CTM synchrony nor a reproduced latent-CoT curriculum |

The memory and cognition chapters explain the mechanisms and costs. We do not
stack every paper's full architecture into the first model. The common ports,
state ownership and candidate interface allow meaningful incremental experiments
within one system instead of disconnected implementations.

## 3. Why the pretrained-model decision is sound, and its limit

The pretrained model supplies expensive language/vision representations. The
project learns the interface between those representations, external experience
and internal adaptive state. This is a reasonable engineering strategy for the
available hardware. It is not a guarantee that a small number of examples can
teach arbitrary new memory or reasoning behavior.

MemT-1.0 is particularly relevant, but its reported 34B-token pipeline and the
HOPE adaptation scale are evidence that architecture alignment can require
substantial training. They are not a local budget recommendation. Our first
question is narrower: whether a bounded trained interface improves use of real
experience on new tasks while retaining native capability. Proposed small
dimensions reduce engineering cost; useful capacity must be measured.

## 4. Model-family similarity does not imply identical backbones

The pinned [MiniCPM5 configuration](https://huggingface.co/openbmb/MiniCPM5-2B/blob/f97400052a43d642bbc6e9975e2397e3ae6a6b52/config.json)
identifies a 42-layer width-2048 Llama decoder. The pinned
[MiniCPM-V-4.7-1B configuration](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B/blob/379f120c2ff3588e6e06559ec6b9ec15cc05114e/config.json)
identifies a 24-layer width-1024 hybrid Qwen3.5 text core plus vision. The VLM
requires its own handling of recurrent state, full attention and visual positions.
The [backbone chapter](02_Backbone_Integration.md) includes exact source/code pins.

The small models are valid technology-validation targets. Port schemas, learning
objectives, data lineage and experimental methods can transfer to larger models;
projection weights, memory images and assumptions about every native block
cannot transfer without inspection, alignment and migration.

## 5. Design decisions to revisit with evidence

| Decision now | Reason | Revisit when |
|---|---|---|
| Complete-block residual ports, native operators preserved | Clear null path and minimal native API disruption | A trained pre-attention fusion alternative demonstrates an advantage |
| Two-pass decision schedule | Removes late-feature/early-injection cycles and label leakage | A causal pipelined/cache implementation matches behavior with lower total cost |
| Per-record contextual writer | Source-level invalidation and native-feature alignment | A cheaper encoder preserves task usefulness and compatibility |
| Linear fast-memory reference, shared-depth option | Transparent functional state contract | Nonlinear/GDN-2 variants earn stability and cost through later-query gains |
| Exact ledger plus compact neural state | Addresses both provenance and associative access | No single compact representation is assumed to replace exact source needs |
| Supervised control, then learned selection | Produces interpretable real action/outcome training | Reliable outcomes support bandit/RL selection under a fixed evaluator |
| Frozen deployment epochs | Keeps attribution, rollback and data migration explicit | A separately admitted developmental-base revision is warranted |

## 6. Research inventory and limits

- [Backbone source manifest](../../reports/architecture_research_20261023/backbone_sources.json): model configs, official Transformers source, local runtime hashes and header inventory.
- [Memory source manifest](../../reports/architecture_research_20261023/memory_sources.json): versioned papers and repository/file hashes with inspection scope.
- [TAIS and learning manifest](../../reports/architecture_research_20261023/tais_learning_sources.json): upstream tree/revision, selected source identities and directly reviewed implementation findings.

No third-party model execution, pretrained weight download, training, performance
reproduction or license grant is implied by source inspection. The older R2
archive, experiment reports and model source remain unchanged. Repository terms
vary; this design adopts ideas and explicit contracts, and imports no upstream
implementation into the project's model code.

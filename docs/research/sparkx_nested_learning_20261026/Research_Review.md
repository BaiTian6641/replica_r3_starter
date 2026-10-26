# Primary-source review: what supports the nested learner

**26 October 2026.** Full methods were inspected, using versioned papers and pinned optimizer source. The [evidence manifest](../../../reports/sparkx_nested_learning_20261026/evidence.json) records URLs, versions, inspected sections and SHA-256 hashes. Sixteen papers are considered below. Results are author-reported, not reproduced in this repository. Proposed Spark connections are our design inferences.

## 1. Nested Learning is a design framework; HOPE is a particular implementation

[Nested Learning / HOPE v1](https://arxiv.org/html/2512.24695v1), especially §§3.2–3.3, describes optimization levels through their objectives, context flows and update frequencies. It permits several kinds of knowledge transfer, including conditioning, initialization learning and gradients. §§7.1–7.3 include sequential, nested and independent/headwise continuum-memory arrangements; a serial chain is therefore not mandatory. §8's self-referential memory is more specific: adaptive memory components generate update targets, with meta-learned initial states and a stated update algorithm. §9.1's pretrained Llama adaptation includes 15B continued-pretraining tokens. Its benchmarks do not establish indefinitely reliable audiovisual learning at a small-device cost.

**Design consequence:** parallel plastic branches can be a legitimate inspiration. We must name their objectives and transfer mechanisms, and establish future learning empirically. Calling all modules “HOPE” would erase material differences. The product target is a **nested-learning-inspired native learner**; a faithful HOPE reproduction remains a separate candidate. Multiple timers alone do not justify our product claim, although the paper's framework is broader than differentiable meta-learning alone.

## 2. Associative memory needs a complete update specification

| Source | Verified mechanism and limit | Spark decision |
|---|---|---|
| [Titans v1](https://arxiv.org/html/2501.00663v1) | Gradient-based neural memory with surprise, momentum and forgetting; several attention/memory compositions. Its persistent tokens and adaptive memory are different mechanisms. | Keep a simple delta-rule control before deep neural memory. Do not equate persistent task tokens with user-specific knowledge accumulated over sessions. |
| [MIRAS v1](https://arxiv.org/html/2504.13173v1) | Separates the memory architecture, associative objective, retention mechanism and optimization rule. | Every `F` candidate must declare all four. A named memory layer with an unspecified loss or retention rule is incomplete. |
| [ATLAS v1](https://arxiv.org/html/2505.23735v1) | Extends memory fitting to recent context and studies capacity/feature maps. OmegaNet and ATLAS differ; the latter uses a Muon-based update. | Test bounded recent-observation replay as an independent hypothesis. An AdamW or delta-rule candidate is not a faithful ATLAS implementation. No new Muon run is proposed under the repository policy. |
| [TTT layers v4](https://arxiv.org/html/2407.04620v4) | Treats recurrent state as a learned model, with an inner reconstruction task and outer task training. §§2.2–2.5 expose gradients through the inner update and the minibatch/parallelization tradeoff. | Train the write representation for later useful reads. Chunked writes change the algorithm and require identical training/inference timing. Linear complexity alone does not establish low wall-clock cost. |

The practical connection is a small, bounded memory trained to compress useful observations. This does not guarantee arbitrary exact recall. Episodic records remain the reference for names, numbers, provenance and corrections; plastic parameters are tested for reusable relations and skills. Compression should earn its place against that reference at measured cost.

## 3. Intrinsic integration and sharing across depth

[MemT-1.0 v2](https://arxiv.org/html/2610.26825v2) supports integrating memory attention into backbone computation, with fixed hyper-memory machinery and local dynamic matrices. Its SGDN-2 update and **dense** Mixture-of-Storage are specific algorithms; the paper reports a 34B-token pipeline. Our post-block control, sparse top-two access, cross-session persistence and plastic MLP bank are deviations, not supplied capabilities.

[Universal Test-Time Training v1](https://arxiv.org/html/2610.05484v1) directly addresses memory shared across layers. All participating layers read chunk-start state; their contributions combine into one boundary update. Historical cross-depth gradient paths are studied explicitly. The paper separates total, reachable and active state and evaluates matched controls. Its particular optimizer/routing implementation is not our baseline.

**Design consequence:** more reader ports need not imply a separate memory copy or a write at every layer. Our reference uses immutable reads, aggregates eligible write contributions, and performs one state transition per observation/chunk. Compare shared and private memories at both matched storage and matched active work. Do not attribute a gain from additional reads or wider capacity solely to sharing. Preserve a gradient path from a later shallow read to an earlier deep write in offline training where that path is claimed.

## 4. Training a representation to support later learning

| Source | What the methods establish | Connection and boundary |
|---|---|---|
| [OML v2](https://arxiv.org/pdf/1905.12588v2) | Algorithm 2 trains a representation for sequential online updates and later prediction; experiments include regression and incremental character classification. | Use chronological teaching and retention objectives. A representation good for ordinary classification is not automatically good for repeated adaptation. Scale and modalities differ substantially. |
| [ANML v2](https://arxiv.org/pdf/2002.09571v2) | Meta-learns neuromodulation and prediction machinery; its meta-test protocol fixes the modulation network and updates the prediction head. | A learned plasticity controller may be useful after fixed-policy adaptation works. This does not support claims of subjective emotion or an autonomous cognitive system. |
| [La-MAML v2](https://arxiv.org/pdf/2007.13904v2) | Combines episodic replay and a meta-objective; §4.2 learns learning rates from the relationship between current and previous-task gradients. | Useful motivation for adaptive update allocation. Per-parameter rates and continual meta-updates add state and cost; they are a later experiment, not free functionality. |
| [MAML v3](https://arxiv.org/pdf/1703.03400v3) | Optimizes initial parameters for performance after adaptation and discusses a first-order approximation to the meta-gradient. | Separate the desired post-adaptation objective from its gradient estimator. First-order training may be practical, but is not numerically equivalent to exact unrolling. |
| [TTT-E2E v2](https://arxiv.org/html/2512.23675v2) | Learns an initialization for next-token adaptation; changes a restricted MLP subset while retaining static computation. The reported 3B comparison uses 164B training tokens. | Strong precedent for training a language model to benefit from later updates. Its throughput and long-context results do not establish cheap permanent multimodal knowledge acquisition on Spark. |

Our synthesis is deliberately narrower than “the model redesigns itself”: offline training prepares a stable learner; verified later experiences update designated native state and parameters. The first success criterion is retained, source-free transfer to new examples. Learning to improve the update policy is a distinct second claim.

## 5. Vocabulary and Engram are complementary, not interchangeable

[Engram v2](https://arxiv.org/html/2601.07372v2) uses deterministic n-gram hashing into learned static capacity, with contextual gating and local processing. It does not establish that a static lookup table acquires new knowledge online. Its token-based keys do not automatically apply to continuous visual or acoustic events.

Our “dynamic Engram” remains a project term for editable native associative memory. A concept identifier persists independently of its current spelling, pronunciation, phrase-bank row or decoder token position. Canonical base-token sequences can key a static n-gram control. Dynamic visual/audio experience uses representation-versioned event keys. Recycling a vocabulary slot must not recycle knowledge or Adam moments. This separation is essential when testing whether phrase compression merely stores a label or supports semantic learning.

## 6. Multimodal evidence can disappear while answers remain correct

[Hidden Forgetting / RCL v1](https://arxiv.org/html/2607.02020v1) measures changes in evidence reliance using counterfactual channel interventions and adds a reliance-preservation objective. Its limitations acknowledge the need for meaningful channel definitions and extra training-time computation; extending to audio and longer streams is further work.

Our decision is to use **evidence-dependence retention as an evaluation requirement first**. After learning unrelated material, repeat matched image, region, frame-order, sound and synchronization substitutions. Preserve correct behavior under harmless transformations, while requiring appropriate changes when evidence changes. A previously wrong or shortcut-driven reference should not be blindly distilled. Any later reliance-preservation loss needs a qualified reference and its extra forward passes charged. The audio application is our hypothesis, not a result from RCL.

[Is Our Benchmark Enough? v1](https://arxiv.org/html/2606.20961v1) finds that highly separable task domains can make expert routing deceptively easy, and that task order changes conclusions. The scope is a particular continual-MLLM benchmark and methods, not a theorem against shared experts.

Our streams therefore include overlapping concepts, recurring domains, ambiguous task identity, cross-modal instances and several task orders. Report transfer separately from retention. A router that recognizes a domain and selects an isolated expert can be a valuable baseline, but is not sufficient evidence of general experience-based improvement.

## 7. A relevant-looking paper that does not satisfy the architecture requirement

[Learning to Learn from Multimodal Experience / AutoMMemo v1](https://arxiv.org/html/2605.16857v1) searches executable memo programs with update/retrieve operations. Retrieved text and visual references are supplied to an agent. Its limitations describe offline search and the remaining challenge of continual deployment.

This is useful for evaluation ideas and cost accounting, but it is **not evidence of backbone-native plastic learning**. We do not replace the intrinsic architecture with its external program-search mechanism. In particular, a native UAP tensor interface does not convert a prompt-memory agent into a self-learning backbone.

## 8. Optimizer feasibility changes the immediate implementation plan

The repository pins `torch==2.14.1`. The [matching Adam implementation](https://github.com/pytorch/pytorch/blob/v2.14.1/torch/optim/adam.py) explicitly rejects `fused=True` with `differentiable=True`. The [AdamW documentation](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html) distinguishes differentiation through an optimizer step from ordinary parameter gradients.

Consequently, passing ordinary finite-gradient tests is insufficient to claim meta-gradients through updates. The initial implementable design uses native fused AdamW steps plus an explicitly declared first-order meta-gradient for `P` initialization. The functional `F` recurrence can retain its own differentiable outer path. Exact differentiable AdamW is a separately specified mathematical/reference path, requiring numerical parity, higher-order gradient verification and repository protocol qualification before model training. No optimizer-policy exception or backend support is assumed.

The Adam moments also invalidate the earlier small resident-memory illustration: the 16-expert `P` bank alone needs 48 MiB of FP32 weights and 96 MiB of moments. We retain an honest larger resource candidate rather than omitting the optimizer state.

## 9. Overall assessment

Evidence is strongest for the individual ideas: intrinsic associative memory, training for adaptation, selective plasticity and useful update schedules. Evidence is much weaker for their combined long-running audiovisual behavior, corrections after compression, persistent cross-session learning, and recursive policy improvement at the proposed cost.

That combination is the research contribution we must demonstrate. The [learning contract](Nested_Learning_Contract.md) makes it executable in principle; the [validation plan](Validation_and_Open_Issues.md) prevents a successful component or a low training loss from being mistaken for the completed product.

# Efficient LoRA research for persistent learning

The best method depends on whether the constraint is trainable state, saved activations, initialization cost, inference latency or retention. A small trainable-parameter count measures only one of these. For Spark, start with a reproducible low-rank baseline and add complexity only when it improves a measured constraint.

For a frozen linear weight `W[out, in]`, ordinary LoRA uses `W + s BA`, where `A[r, in]` and `B[out, r]` contain `r(in + out)` parameters. It still evaluates the frozen backbone. Training an earlier vision projector or memory writer also requires gradients through intervening frozen layers. Setting their weights to `requires_grad=False` does not permit detaching the hidden-state path.

## Evidence quality and comparison policy

**Learning Rate Matters**, revised in 2026, compares tuned LoRA with several variants across models and math/code tasks. Its deeper experiments include PiSSA, MiLoRA, InitAB and DoRA; they do not cover every method in this report. Results often become close with method-specific learning-rate searches. This supports equal tuning budgets, rather than either declaring vanilla universally best or accepting every new headline at face value. [Paper v2](https://arxiv.org/html/2602.04998v2).

The proposed comparison gives each method the same number of validation trials, the same observations and replay, and an explicit time budget. Record initialization, decomposition, compilation, failed trials, training, validation and export time. Compare both equal target/rank configurations and configurations matched by actual parameter or memory budget; they answer different questions.

## Methods that deserve early evaluation

| Method | Mechanism and evidence | Proposed role |
|---|---|---|
| Standard LoRA | Two trainable factors with a zero initial update | Reference at ranks 8 and 16 |
| rsLoRA | Scales updates with `alpha / sqrt(r)` instead of `alpha / r` | Low-complexity scaling control, especially when comparing ranks |
| LoRA+ | Gives the two factors different learning rates | Small optimizer ablation after tuning the baseline |
| LoRA-FA | Freezes A, trains B and uses a corrected optimizer | First activation-memory alternative |
| EVA | Initializes from activation statistics and can redistribute rank | Optional offline initialization study |
| CARE-LoRA | Compresses saved activations while training both factors | Longer-batch activation study |
| MatryoshkaLoRA | Trains an ordered adapter representation usable at different ranks | Later variable-compute adapter study |
| DoRA | Separately adapts weight magnitude and direction | Quality control when the simpler methods plateau |

**rsLoRA** provides a rank-dependent scaling argument and evidence for more stable high-rank adaptation. It adds little implementation complexity, but does not imply that a particular alpha or learning rate is optimal for Spark. Export the exact scaling convention with every adapter. [A Rank Stabilization Scaling Factor v1](https://arxiv.org/html/2312.03732v1).

**LoRA+** changes the relative learning rates of A and B. Its preferred ratio depends on model, initialization and task; the paper does not support universally setting 16. Spark can test ratios 1, 4 and 16 within a fixed search budget. This changes optimization without adding inference matrices. [LoRA+ v2](https://arxiv.org/html/2402.12354v2).

**LoRA-FA** fixes A and trains B, reducing the need to retain the full input activation for that adapter branch. Its optimizer includes a correction based on the fixed projection; simply freezing A is a different baseline. The fixed subspace may limit repeated adaptation to new skills. Lower activation storage also leaves backbone, attention, optimizer and upstream-gradient costs in place. [LoRA-FA v3](https://arxiv.org/html/2308.03303v3), [pinned PEFT optimizer](https://github.com/huggingface/peft/blob/7d6315ae910bf62b6044a7ca917bccf5881f5e43/src/peft/optimizers/lorafa.py).

**EVA** uses activation statistics to initialize A and can allocate ranks according to explained variance. It preserves a zero initial adapter output through B. Explained variance is a useful allocation proxy, not proof of future task utility. Spark's projection widths differ, so an equal sum of ranks is not necessarily an equal parameter budget. Calibration should include training and retention data, exclude held-out tests, and run once per consolidation campaign. [EVA v5](https://arxiv.org/html/2410.07170v5).

**CARE-LoRA**, July 2026, stores a compressed activation and a small batch-specific reconstruction map. It retains exact forward computation, B gradients and input gradients in its formulation, while approximating the A gradient. This makes it particularly relevant when earlier native modules need credit assignment. Its reconstruction solve and storage can outweigh savings for very short batches. [CARE-LoRA v1](https://arxiv.org/html/2607.11940v1), [pinned implementation](https://github.com/fishandyu/CARE-LoRA/blob/f60d5f3f70f80451e4a4851dcbb945e5b9e4d574/peft/src/peft/tuners/lora/layer.py).

**MatryoshkaLoRA**, May 2026, inserts a fixed diagonal weighting between factors during training to obtain useful nested sub-ranks. The inspected implementation uses this weighted update during training and a selected factor prefix during evaluation. Arbitrarily truncating an ordinarily trained LoRA is not equivalent. The useful question is whether a learned rank choice improves task quality per unit cost after accounting for the unchanged backbone. [Paper v1](https://arxiv.org/html/2605.07850v1), [pinned adapter](https://github.com/IST-DASLab/MatryoshkaLoRA/blob/66dd4641d27d8891ab0737549a4d225c96abd1dd/adapters/matryoshka.py).

**DoRA** separates magnitude from direction and applies a low-rank update to the latter. A fixed trained adapter can be merged, but its unmerged normalization and possible weight temporaries matter for frequent online updates. The paper's detached-normalization memory improvement is relative to its fuller formulation, not a general guarantee of lower memory than vanilla LoRA. [DoRA v6](https://arxiv.org/html/2402.09353v6).

## Initialization and compression methods for offline consolidation

| Method | What to retain from the research | Why it is not the first online update rule |
|---|---|---|
| PiSSA | Train a principal low-rank component and freeze the residual | Changes the frozen base representation; decomposition and export compatibility matter |
| CorDA++ | Context-oriented decomposition and allocation, including a preservation-oriented mode | Covariance and inverse/SVD preparation can be expensive |
| LoRA-One | Initialize using one full-gradient decomposition | Obtaining the gradient and decomposing it still costs memory and time |
| LoRAct | Compress saved activations and reconstruct them for backward computation | Approximate gradients and decomposition overhead need task-specific checks |
| QLoRA | Frozen low-bit base plus higher-precision adapters | Backend and quantized backward support must be qualified |
| LoftQ | Jointly choose quantization and low-rank initialization | Creates a new quantized base/adapter pair |

**PiSSA** decomposes the original weight and trains a selected principal component over a residual base. If converted back to an adapter over the original base, the update is the difference between final and initial low-rank products; its rank can be as high as `2r`. “Same rank” and “same base” must therefore be checked at export. [PiSSA v4](https://arxiv.org/html/2404.02948v4).

**CorDA++** develops context-oriented decomposition with multiple covariance choices and dynamic allocation. Its preservation mode protects selected dominant components while adapting another subspace. That is a relevant retention hypothesis, not a guarantee against forgetting. A covariance over a 10,240-wide Spark input alone occupies 400 MiB at FP32; preparation should process modules sequentially. [CorDA++ v1](https://arxiv.org/html/2506.13187v1).

**LoRA-One** uses a full-gradient-derived initialization. The main algorithm in the inspected version uses AdamW; a separately described variant should not be silently substituted. Its initialization may improve optimization, but “one gradient” still includes a backward pass and potentially large dense tensors. The initial behavior, residual handling and exported base identity need explicit checks. [LoRA-One v3](https://arxiv.org/html/2502.01235v3).

**LoRAct** approximates saved activations with low-rank representations. Forward computation remains intact, while backward computation uses reconstructed activations. The benefit depends on activation size and the decomposition method. For joint vision/memory training, inspect which gradients become approximate; its tradeoff is not identical to CARE-LoRA's localized A-gradient approximation. [LoRAct v1](https://arxiv.org/html/2509.23472v1).

**QLoRA** combines a frozen 4-bit base, NF4, double quantization and paged optimizer support. Stored weights and computation dtype differ. Its original paging mechanism uses NVIDIA unified memory; it is not an MPS memory solution. Spark's official FP8 compressed-tensors release is a different format and requires its own qualification. [QLoRA v1](https://arxiv.org/html/2305.14314v1).

**LoftQ** alternates quantization and low-rank approximation to reduce the initial discrepancy from the original weights. It is appropriate for a separately prepared quantized adaptation branch. It does not justify treating the official FP8 release as interchangeable with a newly quantized BF16 base. [LoftQ v4](https://arxiv.org/html/2310.08659v4).

## Very small updates and conditional activation

**TinyLoRA** trains a few coefficients over fixed projected update bases, sometimes shared across modules. Its reported 13-parameter reasoning result is narrow evidence about a particular learning setup, not evidence that 13 scalars can absorb arbitrary new knowledge. Fixed bases, model execution and reinforcement-learning rollouts still cost resources. It is a candidate for compact behavioral modulation after a stronger skill-learning baseline exists. [Learning to Reason in 13 Parameters v1](https://arxiv.org/html/2602.04118v1).

**BoostLoRA** applies successive tiny updates to hard examples and changes the available update subspace between rounds. Merging enables low inference overhead, but changes the base checkpoint; cumulative dense updates and decompositions need accounting. The sum of T rank-r updates has rank at most `rT`, not necessarily exactly `rT`. Treat this as an offline consolidation experiment. [BoostLoRA v1](https://arxiv.org/html/2604.27308v1).

**LLMBO**, October 2026, uses Bayesian optimization over a very small adapter coefficient space, evaluating generated outcomes without backpropagation. This can be useful when a backend cannot support the required backward path, but many candidate generations can dominate cost. Its optimization loop does not establish a model-native learning policy. A resulting adapter can live inside Spark; a native controller that learns when and how to change it remains separate research. [LLMBO v1](https://arxiv.org/html/2610.18504v1).

**Activated LoRA**, or aLoRA, applies adaptation only after an invocation boundary, allowing compatible base-prefix KV reuse. This addresses serving cost rather than training activation storage. Prefix reuse fails when tokenization, embeddings, prefix conditioning or relevant memory changes. Adapted suffix caches are not interchangeable with base caches. Conditional activation also prevents replacing the whole behavior with one permanently merged weight. [aLoRA v5](https://arxiv.org/html/2504.12397v5).

## Combining methods without losing the experiment

Start with independent comparisons. Scaling and optimizer changes are easier to combine than changes to weight decomposition or custom backward computation, but every combination still needs a separate artifact identity and validation. For example, rsLoRA changes the scale used by a LoRA-FA correction; passing an unrelated alpha to the optimizer can invalidate the intended update. CARE-LoRA and LoRA-FA remove different gradient requirements, so stacking them is not the natural first experiment.

Train newly introduced small readers, writers and projectors directly when that is cheaper and clearer than wrapping them in another adapter. Use LoRA where preserving the large pretrained matrix is valuable. Keep at most a small bounded set of active skill adapters; creating an adapter for every fact transfers the memory-capacity problem into weights and routing.

Generated adapters from the Dynamic Engram plan and gradient-trained adapters have different update mechanisms. The former require training a native generator that produces useful factors; the latter require an optimization rule and verified training examples. They may share target ports and serialized factor shapes, but their learning claims must be evaluated independently.

For functional appraisal or personality modulation, test a tiny coefficient controller only after its neutral counterpart works. Emotional salience may alter attention or replay priority within a bounded budget. It should not be treated as evidence that an observation is true or a reason to permanently rewrite factual weights. The personality description remains an input to later training, rather than a specification invented here.

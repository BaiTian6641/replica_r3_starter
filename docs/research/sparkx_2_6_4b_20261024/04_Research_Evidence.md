# Research evidence, comparisons and decisions

Literature research cutoff: **24 October 2026**; final repository/FP8 reconciliation: **25 October**. Sources below are primary papers, author project pages or official model repositories. The review used web search plus direct reading of relevant sources; it did not reproduce their experiments. Spark configuration/implementation/template/index were additionally fetched at a pinned revision and inspected as text. For most secondary candidate methods, the evidence scope is the paper abstract/release rather than a full implementation audit. Full-text mechanism checks focused on Titans and Nested Learning.

Evidence has three levels throughout this report set: **observed locally** (code/tests/recorded results with stated review limits), **reported externally** (paper or vendor evidence), and **proposed here** (the Spark integration and numeric experiment choices). No published source reviewed here establishes the complete combination of Spark, vision, recurrent core, continual learning and personality.

## Backbone and fast adaptation

| Primary source | What it contributes | Decision and limit |
|---|---|---|
| [Spark-X2.6 official release](https://github.com/XHToken/Spark-X2.6) | The release states that X2.6 retains X2.5 architecture and improves post-training; configurable thinking levels | Use the requested pretrained model. Treat benchmark charts as vendor claims; evaluate local tasks at matched reasoning budgets |
| [Pinned model configuration](https://huggingface.co/XHToken/Spark-X2.6-4B/blob/66e3ae7bee54e0a523d692cb8ef841f35409807b/config.json) and [implementation](https://huggingface.co/XHToken/Spark-X2.6-4B/blob/66e3ae7bee54e0a523d692cb8ef841f35409807b/modeling_spark.py) | Hybrid sliding/full attention, fused projections, head gates and distinct rotary settings | Build a dedicated native adapter. Metadata/source inspection is not runtime parity |
| [Official Spark MLX implementation](https://github.com/XHToken/Spark-MLX-LLM) | Apple Silicon inference reference | Useful cross-check; does not establish the repository's PyTorch training or adaptive-memory support |
| [Pinned official FP8 configuration](https://huggingface.co/XHToken/Spark-X2.6-4B-FP8/blob/0318d6f6ab070c7b5b43052db571489b2508006c/config.json) | The current repository's additional adaptation-base candidate with specified weight/activation quantization | Independently fetched configuration; no native kernel/backward support established. Preserve independent BF16 and named derivative controls |
| [TTT, v4](https://arxiv.org/abs/2407.04620v4) | Linear/MLP hidden-state models updated through a self-supervised inner objective | Study functional inner learning; expressive state still requires appropriate outer training and can carry I/O cost |
| [Titans, v1](https://arxiv.org/html/2501.00663v1) | Deep associative memory, momentum/forgetting, and alternative attention-memory compositions | Add a small gated nonlinear branch as a controlled retrofit hypothesis. Fixed pretrained Spark blocks plus a new branch are not a reproduction of a trained Titans architecture |
| [Gated DeltaNet-2, v1](https://arxiv.org/abs/2605.22791v1) | Separates channel-wise erase and write in recurrent memory | Preserve efficient linear controls. The repository's SGDN-2 path is a specific implementation to validate, not interchangeable with every published GDN-2 variant |
| [Universal Test-Time Training, v1](https://arxiv.org/abs/2610.05484v1) | Shared fast memory across depth/time, compared with layer-private ownership | Test shared versus private state with normalized write aggregation and genuine historical gradients; fixed-budget controls matter |
| [MemT-1.0, v2](https://arxiv.org/abs/2610.26825v2) | A memory-native model program with backbone preparation and explicit continued/post-training; reports 34B training tokens | Strong reason to train memory utility explicitly. Its scale and architecture do not establish a cheap local Spark retrofit; distinguish it from the different “Mem-T” agent-memory work in older repo notes |
| [Nested Learning / HOPE, v1](https://arxiv.org/html/2512.24695v1) | Optimization levels, CMS and self-referential adaptive modules | Start with simpler rate-separated modules; label them HOPE-inspired. The pretrained-model study's 15B continued-pretraining tokens are a scale warning, not our run budget |

The central distinction is between learning **how to encode/read/update** a memory during development and applying that learned update at observation time. A useful inner loss cannot be assumed from adding gradient descent to an arbitrary MLP. The repository's negative learned-memory results make representation and readout supervision the highest-priority local experiments.

“Persistent memory” also has different meanings across papers and software: pretrained task parameters, session fast state, exact records and saved checkpoint images are not the same object. Each proposed module therefore has an explicit owner, update rule, persistence rule and test.

## Visual representation and reasoning

| Primary source | What it contributes | Decision and limit |
|---|---|---|
| [BLIP-2](https://arxiv.org/abs/2301.12597) | A trained bridge between frozen image and language components | Supports separate visual-bridge training. A Q-Former is one choice, not required for the first Spark arm |
| [LLaVA / Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) | Connecting vision and language with visual instruction training | Alignment and instruction training are separate stages. Caption loss alone is not evidence of spatial/temporal reasoning |
| [SigLIP 2](https://arxiv.org/abs/2502.14786) | Visual-language encoders with localization/dense-feature improvements and aspect/resolution variants | Recommended initial pretrained image family. Pin the exact vision tower/processor and compare token compression |
| [DINOv3 author report](https://ai.meta.com/research/publications/dinov3/) | Dense self-supervised image features, including strategies to retain dense quality | Optional teacher/second-tower ablation for geometry and correspondence; it is not automatically language-aligned |
| [Qwen2.5-VL](https://arxiv.org/abs/2502.13923) | Dynamic resolution and explicit temporal modeling | Retain size/coordinate/time metadata. Do not transplant its whole positional scheme into Spark without retraining/parity analysis |
| [Qwen3-VL](https://arxiv.org/abs/2511.21631) | Multi-level visual feature integration, interleaved spatial/temporal positions and timestamp alignment | Makes multi-level visual access and real temporal encoding worth testing. New architecture proposal remains Spark-specific |
| [V-JEPA 2](https://arxiv.org/abs/2506.09985) | Video representation learning and latent prediction/planning research | Optional predictive target/teacher. Language alignment and action-conditioned planning require additional learning |
| [V-JEPA 2.1](https://arxiv.org/abs/2603.14482) | Newer dense spatial/temporal self-supervision over image/video features | Prefer considering it alongside 2.0 for teacher selection; compare actual availability, processor, parameter count and local utility |
| [Visual Program Distillation](https://arxiv.org/abs/2312.03052) | Distilling verified visual/tool reasoning trajectories into a VLM | Useful route to capabilities inside the deployed model. Teacher/program errors need independent filtering |
| [SpatialVLM](https://arxiv.org/abs/2401.12168) | Spatially supervised data generation and learning for visual question answering | Supports explicit geometry supervision; monocular metric estimates still require uncertainty and appropriate scale assumptions |

The practical recommendation is one image tower plus a small temporal learner first. Dual towers, large video encoders and a learned world simulator are competing additions with cost and alignment burdens. Training/evaluation must demonstrate the marginal gain of each before the model becomes too expensive to iterate on.

Spatial capability should be measured through evidence-dependent relations and transformations. Temporal capability should be measured where static content is insufficient. Visual reasoning should continue to use visual states throughout the computation; converting everything into a caption before reasoning is a useful baseline but an unnecessary information bottleneck for the target architecture.

## Semi-cognition and continual improvement

| Primary source | What it contributes | Decision and limit |
|---|---|---|
| [Continuous Thought Machines](https://pub.sakana.ai/ctm/) | Neuron-history computation and synchronization representations across internal ticks | Research alternative to the recurrent workspace, especially active inspection. Current slot recurrence is not CTM synchronization |
| [Tiny Recursive Models](https://arxiv.org/abs/2510.04871) | Small recurrent computation with strong results in specified puzzle settings | Compare recurrence against matched simple controls. Puzzle results do not establish conversational, multimodal or continual-learning transfer |
| [SEAL](https://arxiv.org/abs/2506.10943) | Model-generated adaptation data/directives coupled to real fine-tuning and downstream feedback | Adopt the bounded proposal→actual update→evaluation loop. General self-improvement is not guaranteed by self-generated data |
| [STaR](https://arxiv.org/abs/2203.14465) | Iterative reasoning bootstrapping with correctness-filtered training examples | Use verified practice. A correct answer does not certify all intermediate visual or causal statements |
| [Dark Experience Replay](https://arxiv.org/abs/2004.07211) | Rehearsal with historical output consistency as a continual-learning baseline | Include replay/distillation before complex multi-rate claims; its experiments do not validate our multimodal recipe |
| [Self-Distillation Enables Continual Learning](https://arxiv.org/abs/2601.19897) | Demonstration-conditioned self-teaching on student-policy outputs | A candidate instructed-learning arm; requires sufficient instruction-following/ICL capability |
| [SynthSelf](https://arxiv.org/abs/2610.13113) | Recent study of small-model limits of SDFT and a larger same-family alignment aid | Prevents assuming self-distillation transfers automatically to a 4B model. External alignment is an optional training aid, not required at deployment |
| [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954) | Empirically evaluated software changes and an archive of agent variants | Later code/architecture research arm. Software evolution and neural weight improvement must be reported separately |
| [Intrinsic self-correction study](https://arxiv.org/abs/2310.01798) and [condition-verification study](https://arxiv.org/abs/2405.14092) | Different methods/settings report limitations and improvements in self-correction | Avoid a universal impossibility claim. Use correction as a proposal mechanism, and independently test whether it helped on our tasks |

The repository's semi-cognitive core is a plausible location for learned attention allocation, bounded iteration and improvement selection. It presently lacks evidence that those behaviors generalize. A stronger control policy and a reliable evidence/learning loop matter more than adopting a biologically inspired name. Personality fidelity is a separate empirical question that cannot be answered before the promised source file and appropriate examples arrive.

## Evaluation sources

| Source | Role in the plan |
|---|---|
| [CLEVR](https://arxiv.org/abs/1612.06890) | Controlled compositional spatial questions and structured supervision |
| [GQA](https://arxiv.org/abs/1902.09506) | Real-image relational/compositional questions and bias-aware evaluation |
| [TextVQA](https://arxiv.org/abs/1904.08920) | Reading and reasoning about visual text |
| [BLINK](https://arxiv.org/abs/2404.12390) | Perception, correspondence, depth and multi-view weaknesses not captured by fluent captions |
| [TempCompass](https://arxiv.org/abs/2403.00476) | Temporal distinctions with similar static content |
| [Perception Test](https://arxiv.org/abs/2305.13786) | Video reasoning plus spatial/temporal annotations useful for diagnostic decomposition |
| [VSI-Bench / Thinking in Space](https://arxiv.org/abs/2412.14171) | Spatial understanding and recall across video observations |
| [CLEVRER](https://arxiv.org/abs/1910.01442) | Controlled event prediction and counterfactual reasoning |

Use established benchmarks for external comparison and fresh source-family-disjoint episodes for development claims. Existing benchmarks may have entered pretrained models; benchmark scores alone cannot prove novel learning. Synthetic data provides controllable truth but weak evidence of real-world transfer unless paired with real held-out sources.

## Choices to make after the first measurements

| Decision | Default proposal | Evidence that would change it |
|---|---|---|
| Additive versus replacement backbone modules | Preserve native blocks, add independent gated readers | Recovered pretrained behavior and a clear accuracy/cost advantage after replacement training |
| Linear versus nonlinear fast state | Linear/SGDN-2 reference, then nonlinear candidate | Delayed-use/correction gains at matched state and total compute |
| Prefix versus deeper visual access | Prefix baseline, then selected residual readers | Measured long-range visibility or fine-detail bottleneck |
| Image-only versus dual/video towers | One pretrained image tower + temporal module | Task-specific representation gap that survives adequate data and training |
| Eight versus more workspace slots | Preserve eight-slot reference, test 16 | Core-specific success gain over compute-matched control |
| Fixed versus learned learning selector | Fixed verified replay first | Independent better acquisition/retention per total cost |
| SFT+replay versus SDFT | SFT+replay as a transparent baseline | Reliable demonstration conditioning and improved transfer/retention |
| HOPE-inspired versus faithful HOPE retrofit | Small declared multi-rate branch | Funding/capacity for substantial continued pretraining plus mechanism-faithful implementation |
| Added-module size | Confirmed 4B-class Spark backbone plus additional vision, memory and inter-layer learning modules | Measured capacity and accuracy/cost determine module sizes; the total model has no user-specified parameter cap |

HOPE / Nested Learning and the backbone-plus-additional-modules parameter scope were confirmed by the user on 25 October 2026. Unresolved project inputs are the personality description, preferred first application/domain, training-data access and acceptable compute/time budget. The proposed defaults permit interface and pilot design now. Those inputs should determine expansion, rather than assuming general visual intelligence or open-ended recursive improvement can be obtained with a small unspecified fine-tune.

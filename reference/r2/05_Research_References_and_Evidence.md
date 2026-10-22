# 05 — Research references and evidence register

**TAIS Obsidian R2 · reviewed 22 October 2026**

The combined R2 framework is a proposed synthesis. A paper’s success with one mechanism does not validate the whole framework, establish consciousness, or guarantee indefinite improvement. Author repositories are implementation references, not tested dependencies of this package. Pin a compatible immutable revision before execution; a `main` URL is only a discovery link.

## 1. Memory, adaptation, and ownership

<a id="r01"></a>
### R01 — Engram
Xin Cheng et al. (2026). *Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models*. arXiv:2601.07372.

[Paper](https://arxiv.org/abs/2601.07372) · [Author repository](https://github.com/deepseek-ai/Engram)

**Use:** sparse lookup and context-conditioned fusion, separating storage from expensive computation. **Boundary:** the original learned n-gram tables are not a general, editable autobiographical archive. The R2 dynamic writer/record interface is an extension, not a published Engram variant. The released demonstration is not a four-backbone retrofit trainer.

<a id="r02"></a>
### R02 — Titans
Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. *Titans: Learning to Memorize at Test Time*. arXiv:2501.00663; first submitted 31 December 2024.

[Paper](https://arxiv.org/abs/2501.00663) · [Unofficial implementation carried forward from R1](https://github.com/lucidrains/titans-pytorch)

**Use:** trained neural memory, online updates, retention/forgetting mechanisms, and alternative attention–memory interfaces. **Boundary:** a small R2 fast-state branch is Titans-inspired unless it reproduces the paper’s equations and protocol. Neural state is not the sole authoritative personal-history store. The linked code is not an author release.

<a id="r03"></a>
### R03 — Nested Learning / HOPE
Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, and Vahab Mirrokni (2025). *Nested Learning: The Illusion of Deep Learning Architectures*. arXiv:2512.24695.

[Paper](https://arxiv.org/abs/2512.24695) · [Official research explanation](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/) · [Unofficial reduced-scale reference carried forward from R1](https://github.com/smallhours19/nested-learning-hope)

**Use:** explicitly defined learning levels, update frequencies, objectives, and context flows. **Boundary:** periodic database maintenance is not HOPE. R2’s multi-rate side branch must be labelled a synthesis. No author-maintained HOPE trainer was verified for this package; the linked reproduction is unofficial, not evidence of exact fidelity.

<a id="r04"></a>
### R04 — Universal Test-Time Training
Zefan Cai, Qinzhe Hu, Ziqiao Ma, Hao Tan, and Junjie Hu (2026). *Universal Test-Time Training*. arXiv:2610.05484.

[Paper](https://arxiv.org/abs/2610.05484) · [Author repository](https://github.com/Zefan-Cai/uTTT)

**Use:** shared fast-state ownership across depth; private/shared comparisons. **Boundary:** shared adaptive memory is not the pretrained MoE expert bank. At review, the repository describes language-model uTTT-MoE release, not a complete released dense-LM version or pretrained checkpoints for the four target models. Respect chunk read/write causality and per-session isolation.

## 2. Reasoning and cognitive computation

<a id="r05"></a>
### R05 — Continuous Thought Machines
Luke Darlow et al. (2025). *Continuous Thought Machines*. arXiv:2505.05522.

[Paper](https://arxiv.org/abs/2505.05522) · [Author repository](https://github.com/SakanaAI/continuous-thought-machines)

**Use:** internal computation over time, temporal neural processing, and synchronization as an experimentally testable representation. **Boundary:** RoPE over tick indices is not equivalent to CTM synchronization. The demonstrated tasks do not establish that adding a CTM-like branch improves a pretrained 35B language model. R2 includes a simpler recurrent-workspace control and an optional CTM-inspired comparison.

<a id="r06"></a>
### R06 — Coconut
Shibo Hao et al. (2024). *Training Large Language Models to Reason in a Continuous Latent Space*. arXiv:2412.06769.

[Paper](https://arxiv.org/abs/2412.06769) · [Author repository](https://github.com/facebookresearch/coconut)

**Use:** a precedent for training latent computation and a curriculum that changes how intermediate reasoning is represented. **Boundary:** continuous hidden states are not automatically better than textual reasoning; match inference resources, training data, and task families. Do not call R2’s slot workspace a Coconut reproduction.

## 3. Learning how to improve

<a id="r07"></a>
### R07 — SEAL
Adam Zweiger et al. (2025). *Self-Adapting Language Models*. arXiv:2506.10943.

[Paper](https://arxiv.org/abs/2506.10943) · [Author repository](https://github.com/Continual-Intelligence/SEAL)

**Use:** self-generated learning material/update directives, evaluated by subsequent performance after adaptation. It motivates a learned proposal/compiler policy, not just passive logging. **Boundary:** tested adaptation tasks do not prove unconstrained, indefinite self-improvement. R2 retains independent evaluation and bounded update authority.

<a id="r08"></a>
### R08 — Absolute Zero
Andrew Zhao et al. (2025). *Absolute Zero: Reinforced Self-play Reasoning with Zero Data*. arXiv:2505.03335.

[Paper](https://arxiv.org/abs/2505.03335) · [Author repository](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner)

**Use:** self-proposed tasks and solutions grounded in an executable environment. **Boundary:** no external task-training data does not mean no pretrained knowledge, environmental structure, or verifier. R2 uses small deterministic domains first; the model cannot edit the hidden evaluation oracle.

<a id="r09"></a>
### R09 — Limits of untrained intrinsic self-correction
Jie Huang et al. (ICLR 2024). *Large Language Models Cannot Self-Correct Reasoning Yet*. arXiv:2310.01798.

[Paper](https://arxiv.org/abs/2310.01798)

**Use:** a warning against assuming that an untrained model’s repeated reconsideration is reliable evidence of correctness. **Boundary:** the result is not a theorem that all trained self-correction is impossible. Read with R10; R2 permits self-correction while evaluating it independently.

<a id="r10"></a>
### R10 — SCoRe
Aviral Kumar et al. (2024). *Training Language Models to Self-Correct via Reinforcement Learning*. arXiv:2409.12917.

[Paper](https://arxiv.org/abs/2409.12917)

**Use:** evidence that deliberate training can improve self-correction in the studied setting. **Boundary:** a self-correction capability is not an infallible verifier or permission to approve its own deployment. R2 distinguishes proposal generation from acceptance authority.

<a id="r11"></a>
### R11 — Voyager
Guanzhi Wang et al. (2023). *Voyager: An Open-Ended Embodied Agent with Large Language Models*. arXiv:2305.16291.

[Paper](https://arxiv.org/abs/2305.16291) · [Author repository](https://github.com/MineDojo/Voyager)

**Use:** curriculum, environment feedback, and a growing executable skill library without treating every improvement as a backbone weight update. **Boundary:** environment-specific skill acquisition is not unlimited general cognition. R2 compares executable procedures, retrieved procedures, and neural adapters rather than assuming one representation always wins.

## 4. Personal replica and individuality

<a id="r12"></a>
### R12 — Interview/self-report-grounded simulations of individuals
Joon Sung Park et al. (2024; revised 28 June 2026). *LLM Agents Grounded in Self-Reports Enable General-Purpose Simulation of Individuals*. arXiv:2411.10109v3. The earlier version was titled *Generative Agent Simulations of 1,000 People*.

[Paper](https://arxiv.org/abs/2411.10109)

**Use:** interview-grounded individual modeling and evaluation against actual human responses, including human test–retest reference behavior. **Boundary:** success on the paper’s selected measures is not a percentage of an entire person reproduced. It does not establish continuity of consciousness, complete autobiography, or fidelity to unseen engineering decisions.

<a id="r13"></a>
### R13 — Persona vectors
Anthropic (2025). *Persona vectors: Monitoring and controlling character traits in language models*.

[Author research page](https://www.anthropic.com/research/persona-vectors)

**Use:** selected trait-monitoring and intervention methods as diagnostics. **Boundary:** a handful of controllable trait directions is not a complete identity. For R2, decisions, values, provenance, temporal change, and general capability need independent tests; vector similarity cannot be the sole identity metric.

<a id="r14"></a>
### R14 — LaMP
*LaMP: When Large Language Models Meet Personalization*.

[Author benchmark page](https://lamp-benchmark.github.io/) · [Author repository](https://github.com/LaMP-Benchmark/LaMP)

**Use:** an external personalization evaluation reference. **Boundary:** generic personalized classification/generation does not replace a sealed source-person decision set. Data licensing, exact task release, and allowed use must be checked before downloading or training.

## 5. Writer–reader precedents

<a id="r15"></a>
### R15 — Memory of Amortized Contexts
*Online Adaptation of Language Models with a Memory of Amortized Contexts*. arXiv:2403.04317.

[Paper](https://arxiv.org/abs/2403.04317) · [Author repository reference](https://github.com/jihoontack/MAC)

**Use:** train a context encoder so new observations become useful modulations through forward computation. **Boundary:** forward-only acquisition after training is not training-free architectural development. This MAC acronym differs from Titans’ memory-as-context variant.

<a id="r16"></a>
### R16 — LongMem
*Augmenting Language Models with Long-Term Memory*. arXiv:2306.07174.

[Paper](https://arxiv.org/abs/2306.07174) · [Author repository reference](https://github.com/Victorwz/LongMem)

**Use:** frozen source representation and a trained memory-reading network. **Boundary:** memory representation compatibility changes when writers/readers/backbones change. R2 requires re-encoding or tested migration, not interchangeability by tensor shape.

## 6. Model sources and engineering

<a id="r17"></a>
### R17 — MiniCPM5-2B
[Official checkpoint](https://huggingface.co/openbmb/MiniCPM5-2B) · [Configuration](https://huggingface.co/openbmb/MiniCPM5-2B/blob/main/config.json) · [OpenBMB repository reference](https://github.com/OpenBMB/MiniCPM)

The card reports 2,516,756,480 total parameters. Count unique loaded parameters, tied groups, and active modalities for the actual experiment. Keep the requested checkpoint rather than silently substituting a Base/SFT sibling.

<a id="r18"></a>
### R18 — MiniCPM-V 4.7
[Official small checkpoint](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B) · [Official MoE checkpoint](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B) · [OpenBMB repository reference](https://github.com/OpenBMB/MiniCPM-V)

Preserve native visual preprocessing, insertion, hybrid-state, and generation semantics. Count the entire selected checkpoint. Text-only training with a frozen/inactive vision encoder must not be reported as full multimodal fine-tuning.

<a id="r19"></a>
### R19 — Gemma 4
[Official model card](https://ai.google.dev/gemma/docs/core/model_card_4) · [Checkpoint reference](https://huggingface.co/google/gemma-4-26B-A4B-it) · [Google PyTorch reference](https://github.com/google/gemma_pytorch)

The roster uses the official **26B-A4B** spelling. Large-model ports need fresh feature/interface training and model-specific validation; small-model memory vectors do not transfer just because their dimensions match.

<a id="r20"></a>
### R20 — PyTorch autograd mechanics
[Official documentation](https://docs.pytorch.org/docs/2.14/notes/autograd.html)

Frozen parameters and no-grad execution are distinct. Intermediate trainable modules can require input gradients through downstream frozen layers. A trainable parameter allowlist must be checked against the actual optimizer and gradient graph.

<a id="r21"></a>
### R21 — PyTorch activation checkpointing
[Official documentation](https://docs.pytorch.org/docs/2.14/checkpoint.html)

Recomputed forwards must not duplicate durable writes or mutate external session state. Return functional state transitions and commit once outside recomputation.

<a id="r22"></a>
### R22 — DeepSpeed ZeRO-Offload
[Official tutorial](https://www.deepspeed.ai/tutorials/zero-offload/)

Reference for offloading optimizer state/computation while retaining full-weight updates. It does not certify any particular new model/fast-state/mixed-GPU configuration. Test the actual chosen integration and measure CPU bandwidth and per-device peaks.

<a id="r23"></a>
### R23 — Executable and formal engineering verifiers
[Verilator author repository](https://github.com/verilator/verilator) · [SymbiYosys author repository](https://github.com/YosysHQ/sby)

Use simulation and supported formal properties as independent checks for digital-engineering tasks. Passing a model-generated testbench alone is insufficient. Keep simulator versions, reset/clock assumptions, reference semantics, test coverage, and timeouts in the task manifest. Formal results prove only the stated properties under their assumptions.

## 7. Uploaded evidence and source mapping

These are project documents, not independent replications. All source filenames and hashes are in `source_manifest.json`. No raw TAIS repository, model checkpoint, or GPU run was executed for this review.

<a id="u01"></a>
**U01 — `TAIS_Obsidian_子系统架构规格(1).md`.** Parts A–H: full-system components, KAL/HRL, write permissions, consolidation and identity/security. Preserve terminology while separating analogy from technical proof.

<a id="u02"></a>
**U02 — `TAIS_Obsidian_思维能力强化_架构设计文档(1).md`.** §§1–2 and §6: goals, workspace, thought core, inquiry links; §8: explicit empirical limits. It establishes intended scope, not demonstrated language-level reasoning superiority.

<a id="u03"></a>
**U03 — `TAIS_Obsidian_主动求知闭环_架构设计文档(1).md`.** §§1–7: active inquiry, external evidence, consolidation and reward; §8 identifies proposed coupling formulas. R2 retains the loop and revises overbroad impossibility/verification claims.

<a id="u04"></a>
**U04 — `动态知识块记忆系统_设计文档(1).md`.** §§3, 10, 14: evidence/compiled forms, persona blocks, lifecycle and self-improvement. R2 treats source text as evidence, not automatically truth, and adds agent/source identity separation.

<a id="u05"></a>
**U05 — `TAIS_Obsidian_0p1B_学术报告(1).md`.** §4 reports memory-gate interference (ordinary context recall 0.688 to 0.250); §5 says thought dynamics did not establish higher accuracy. §3.7 acknowledges entity overlap in the teaching evaluation. All numbers are source-reported, not reproduced here.

<a id="u06"></a>
**U06 — `TAIS_Obsidian_架构接入状态评估(1).md`.** §§1–4 distinguishes pilot components from normal-forward integration and records mock inquiry/certainty paths. Inspect actual code before declaring these repaired.

<a id="u07"></a>
**U07 — R1 package.** `01_Audit_and_Architecture.md`, `02_Implementation_Construction_Plan.md`, `03_Experiments_and_Hardware.md`, `04_References_and_Evidence.md`, and `AGENT_START.md`. Retain useful causal/gradient/state contracts. R2 supersedes memory-dominant ordering, deferred cognition/replica, and late optional full tuning. R1’s reported CPU-reference test status was not rerun for R2.

<a id="u08"></a>
**U08 — `从零构建TAIS-Obsidian_总体实施计划(1).md`.** §1 historical Blackwell workstation; §8A controlled persona evolution. Historical hardware and software versions are not assumed to describe the newly mentioned small workstation.

<a id="u09"></a>
**U09 — `TAIS_Obsidian_接口与实现计划(1).md`.** §§0–4: checkpoint/runtime ownership and TAIS Memory Bus; retained with revised learning authority and state contracts.

<a id="u10"></a>
**U10 — `TAIS_Obsidian_知识内化训练_分析与设计(1).md`.** Support-dependent learning and real-time use. R2 retains episode dependence while adding strict splits and separating persistence from weight consolidation.

<a id="u11"></a>
**U11 — `TAIS_Obsidian_数据集选型(1).md`.** Historical candidate datasets. This revision does not revalidate every dataset license or reproduce the old bulk pretraining mixture; each selected dataset needs a fresh task/license/split review.

<a id="u12"></a>
**U12 — `TAIS_Obsidian_细致框架设计文档(1).md`.** §§12–14: procedural skill growth, reasoning, persona and update timescales. Broad novelty/exclusivity, zero-forgetting, and guaranteed-combination claims are not carried forward as facts.

<a id="u13"></a>
**U13 — `TAIS_Obsidian_部件实现详细计划(1).md`.** Module-level interfaces and milestone heritage. Reuse sound code only after inspection; old checkmarks are not fresh integration tests.

<a id="u14"></a>
**U14 — `D0_0p1B先导实验报告(1).md`.** Early small-model numerical/throughput results and ablations. The document itself says the short-training comparison is not an architecture-wide conclusion.

<a id="u15"></a>
**U15 — `AGENT_PLAN_E+-5_PM-stream(1).md`.** Historical n=5 residual-stream construction experiment; not a requirement to rewrite all four pretrained models.

<a id="u16"></a>
**U16 — `AGENT_PLAN_E+-7_三级注意力栈(1).md`.** Historical attention-stack construction experiment; not a requirement to replace native attention before full-chain evaluation.

## 8. User requirements outside the uploaded design files

The current request restores full-chain semi-cognition and self-improvement and adds a small-workstation full-fine-tuning laboratory. Earlier replica discussion describes documenting ways of thinking, ideas, and selected life experiences, with a parallel copy expected to diverge rather than duplicate consciousness. This is user-provided context, not a finding from a paper or a fully collected personal training corpus.

R2 therefore proposes interview and evaluation schemas without inventing the user’s biography, personality scores, or unprovided choices. Authentic personal records must be supplied/approved separately; no personal dataset is bundled.

## 9. Boundaries of this review

Primary pages above were consulted for R2. Some implementation links are carried forward as author/unofficial references from R1 and are labelled accordingly; their dependency compatibility was not tested. The package neither exhaustively re-verifies the original fifteen-document bibliography nor claims to prove a novel unified cognitive architecture.

The empirical question is whether the full loop improves new-task performance, calibrated control, learning efficiency and personal fidelity under controlled budgets while preserving retention. The documents and static schemas define how to test that question. They do not supply the answer before experiments.

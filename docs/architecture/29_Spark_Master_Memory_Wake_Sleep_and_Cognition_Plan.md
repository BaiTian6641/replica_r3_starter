# 29. Spark X2.6 4B Master Architecture: Memory Backbone, Wake-Sleep Consolidation, Continuum Self-Learning, Semi-Cognition, and Bounded RSI

Date: 2026-11-04  
Status: Confirmed Master Architecture & Staged Execution Plan (Stages 1–6 & Phases 2–4 `V19`–`V24` Natively Qualified on Spark X2.6 4B BF16)

---

## 1. Purpose and Scope

This document unifies the architectural specifications in [04_Semi_Cognition_Learning_and_Replica.md](04_Semi_Cognition_Learning_and_Replica.md), [19_Integrated_Memory_Learning_and_RSI_Design.md](19_Integrated_Memory_Learning_and_RSI_Design.md), [22_State_Mechanism_and_RSI_Contracts.md](22_State_Mechanism_and_RSI_Contracts.md), [23_SparkX_2_6_4B_Final_Product_Blueprint.md](23_SparkX_2_6_4B_Final_Product_Blueprint.md), [24_SparkX_2_6_v5_1_Refinement_Plan.md](24_SparkX_2_6_v5_1_Refinement_Plan.md), and [28_Spark_Persistent_Learner_and_Tiered_Memory.md](28_Spark_Persistent_Learner_and_Tiered_Memory.md) with the empirical findings from native **V16–V18.2** ([20261031_Spark_Paired_Update_V18.md](../experiments/20261031_Spark_Paired_Update_V18.md)) and the literature cross-check ([20261031_Spark_Research_Crosscheck_and_Alignment.md](../experiments/20261031_Spark_Research_Crosscheck_and_Alignment.md)).

Because the complete Spark X2.6 intrinsic-memory and continual-learning system spans memory-layer geometry, online memory formation, offline sleep-time consolidation, multi-timescale plasticity, metacognitive control, and bounded recursive self-improvement (RSI), the program is separated into **five orthogonal architectural categories** and **six gated execution stages**.

At every stage, the project enforces four strict distinctions:
1. **Code implemented** (module exists with deterministic contracts);
2. **CPU verification completed** (unit, property, restoration, and synthetic fixture tests pass);
3. **Native experiment executed** (supervised run on official Spark X2.6 4B BF16 with verified resource, restoration, and ledger receipts);
4. **Behavioral gate passed** (predeclared quantitative margins, accuracy, retention, and calibration gates met without opening sealed evaluation labels).

---

## 2. Empirical Baseline: What Native V16–V18.2 Established

1. **Pretrained Spark X2.6 4B Geometry**:
   - 36 transformer blocks, hidden width `d_model = 2560`, gated GELU MLP intermediate width `10240`, 16 query heads and 4 KV heads with head dimension `256` (`4096` concatenated attention-head width before `o_proj`), hybrid attention schedule with **27 sliding-window-512 blocks** and **9 full-attention blocks** at zero-based indices `[3, 7, 11, 15, 19, 23, 27, 31, 35]`.
2. **V15/V16 Content-Key & Slot Decodability**:
   - Frozen-source features (`FrozenSourceFeatures`) compressed into `4 x 256` ordered contextual slots (`ContextualValueProducerV1`) are linearly decodable on CPU ridge probes (`8/8` world answers recoverable), yet native V16 contextual readout achieved `15/32` world evaluations (`0/8` known–known conflicting pairs both correct) versus `32/32` when exact source text remained in the prompt.
3. **V18.2 (`preparation_attempt04`) Paired-Joint vs Sequential Native Result**:
   - Executing [native_packet.json](../../reports/spark_repairs_v18/preparation_attempt04/native_packet.json) (`SHA-256: 05cf86c28a0838e539256da54b3c87b7c43e67fc27b67a52a6cebbc4e4efed72`) on native BF16 ([result.json](../../reports/spark_repairs_v18/preparation_attempt04/native_bf16_comparison_attempt01/result.json)) proved that **co-optimizing conflicting world pairs (`paired_joint`, `batch_size=2`, 4 steps)** beats sequential single-world updates (`sequential`, `batch_size=1`, 8 steps) at equal world exposure:
     - `mean_final_pair_minimum_margin`: `-0.507505` (`paired_joint`) vs `-1.848542` (`sequential`), yielding `joint_minus_sequential = +1.341037 > 0.0`.
     - Below-baseline guard count: `4/8 <= 4/8`.
     - Complete teacher-forced correctness: `5/8` worlds (`paired_joint`) vs `4/8` (`sequential`) and `2/8` at step-0 baseline, achieving the **first native known–known pair (`1/4`, Pair 2 `v11_fit_08/09`) with both conflicting worlds simultaneously correct** (`+0.335941` and `+0.113274`).
   - **Remaining Bottleneck**: A single exposure per world (`4` paired-joint steps) leaves overall `mean_final_pair_minimum_margin` negative (`-0.507505`). Therefore, **Stage 1 (`V19`)** must jointly test multi-exposure paired-joint convergence (`128` steps / `32` exposures per pair) and frozen-decoder objective alignment (`direct CE` vs `top-64 conditioned-teacher KL`).

---

## 3. Scientific Literature Cross-Check & Mathematical Synthesis

| Research Work | Core Formulation & Architectural Insight | Validation & Adaptation in Spark X2.6 4B |
|---|---|---|
| **MemT-1.0** (`arXiv:2610.26825v2`) | Dedicated memory encoder + gated cross-attention readers injected across transformer layers; compares token-level CE vs top-$K$ logit distillation for memory-slot alignment; introduces experience digestion. | Validates our separated content-key (`512-d`) + ordered contextual value slots (`4 x 256`) and motivates the **Stage-1 V19 matched `direct CE` vs `top-64 teacher KL` comparison** and **Stage-2 site expansion (`2 -> 9 -> 36`)**. |
| **ICAE** (`arXiv:2307.06945v4`) & **LongMem** (`arXiv:2306.07174v1`) | Separates slot autoencoding/compression from frozen-decoder continuation alignment (`ICAE`); caches frozen-backbone memory banks while training residual SideNet readers (`LongMem`). | Explains why linearly decodable slots (`8/8` ridge probe) do not automatically steer a frozen decoder (`15/32` in V16): the producer-reader pair must align with the frozen Spark suffix distribution. |
| **Titans** (`arXiv:2501.00663`) & **Titan-Llama** (`arXiv:2610.19210v1`) | Updates fast neural memory $M_t$ using **momentary surprise** $\nabla \ell(M_{t-1}; x_t)$, **surprise momentum** $S_t = \eta_t S_{t-1} - \theta_t \nabla \ell(M_{t-1}; x_t)$, and **adaptive forgetting (weight decay)** $M_t = (1 - \alpha_t) M_{t-1} + S_t$; `Titan-Llama` retrofits a frozen Llama backbone with staged memory training. | Upgrades our **Category B `S2b` (`ChunkSession` fast associative memory `F`)**: at each causal chunk barrier, `F` updates using data-dependent surprise $\nabla \ell$, momentum $\eta_t \in [0, 1]$, step gate $\theta_t$, and retention decay $\alpha_t \in [0, 1]$ while preserving strict intra-chunk prefix causality. |
| **uTTT** (`arXiv:2610.05484v1`) | Unified Test-Time Training with shared fast state updated in parallel chunks via task-loss gradients at matched active compute. | Validates our **causal chunk-start/chunk-barrier `S2b` contract** (`prefix_token_equality`) and our **Category C `S3` functional one-step writer** evaluated on distinct future queries at equal active expert count (`1` active expert). |
| **HOPE / Nested Learning** (`arXiv:2512.24695v1`) | Continuum Memory System (`CMS`) with nested optimization levels and distinct update clocks ($f_{\text{fast}}, f_{\text{slow}}$) plus surprise-prioritized replay. | Validates **Category C `S5` (`ContinuumMemoryCellV19`)**: fast memory `F`/`Pf` updates at chunk/episode barriers while slow memory `Ps` consolidates across episodes/sleep windows, evaluated against a **matched single-clock control** at identical parameter count. |
| **Sleep-Time Compute** (`arXiv:2504.13171`) & **TAIS CA1/Sleep** ([Doc 04](04_Semi_Cognition_Learning_and_Replica.md), [Doc 24 §7](24_SparkX_2_6_v5_1_Refinement_Plan.md)) | Decouples low-latency wake-phase serving from asynchronous offline sleep-time compute that reorganizes raw interaction logs, synthesizes connections, and distills context into persistent state. | Directly defines **Category B Two-Phase Wake-Sleep Consolidation**: online wake turns only pin authority (`S2a`), update fast chunk memory (`F`), and record verified episodes (`E`/`E2`); offline **Sleep-Time** runs `CA1` eligibility gating, age-balanced replay, and conditioned-teacher forward KL (`KL(q(Y \mid X, M_{\text{ref}}) \parallel p(Y \mid X, P))`) / verified `CE` distillation into `Pf -> Ps`. |
| **SEAL** (`arXiv:2506.10943v2`) | Self-Adapting LLMs generate structured self-edits and training directives that are applied and filtered by downstream task verification. | Validates **Category E (`LearningDirector` + `C0–C5` gates)**: candidate self-edits (`record`, `procedure`, or `neural P candidate`) are trained in isolated branches and promoted only if they pass unseen-task transfer, retention, and identity invariance. |

---

## 4. Five Architectural Categories

### 4.1 Category A — Intrinsic Memory-Layer Backbone (`E2` & `Mkv` on Spark X2.6 4B)

1. **Stage-1 2-Site Calibration (`V19` at sites `[3, 7]`)**:
   - Uses frozen official Spark X2.6 4B BF16, `FrozenSourceFeatures`, `ContextualValueProducerV1` (`4 x 256` slots + `512-d` content keys), and bias-free `SlotMemoryReaderV1` (`zero-initialized` output projection + learned scalar gate) at blocks `[3, 7]`.
   - Compares **128 paired-joint steps (`batch_size=2`, 32 exposures per pair)** under:
     - **Arm 1 (`direct_answer_plus_eos_ce`)**: $\mathcal{L}_{\text{CE}} = -\frac{1}{|T|} \sum_{t \in T} \log p_\theta(y_t \mid y_{<t}, X, M_{\text{slots}})$.
     - **Arm 2 (`top_64_teacher_kl`)**: $\mathcal{L}_{\text{KL}} = \tau^2 \frac{1}{|T|} \sum_{t \in T} \sum_{v \in \text{TopK}(q_t, 64)} q_t(v) \left(\log q_t(v) - \log \hat{p}_t(v)\right)$, where $q_t$ is the frozen full-text teacher distribution (qualified by `evaluate_exact_text_competence`) with the gold token guaranteed in the support.
2. **Stage-2 Backbone Expansion (`2 -> 9 -> 36` sites, Reader Sharing, and Placement)**:
   - **Site Density Progression**:
     - `two`: `[3, 7]` (early full-attention anchor pair).
     - `full_attention_nine` (**Primary Stage-2 Target**): `[3, 7, 11, 15, 19, 23, 27, 31, 35]` (all 9 global full-attention blocks in Spark X2.6, avoiding sliding-window-512 truncation).
     - `all`: `[0..35]` (all 36 hybrid blocks, evaluated only if `full_attention_nine` passes and marginal compute is justified).
   - **Reader Parameterization Regimes**:
     - `independent`: 1 distinct `SlotMemoryReaderV1` per active site (`1,441,792` params/site at width `2560 -> 256 -> 2560`).
     - `matched`: Depth-grouped shared canonical readers (`2` physical reader groups shared across early vs late full-attention blocks, holding reader parameter count constant at `2,883,584` across `2`, `9`, and `36` sites).
   - **Injection Placement**:
     - `block_input_query_to_post_block_residual` (`post-block`, current default): queries from block input, injects residual delta at block output.
     - `block_input_query_to_pre_mlp_residual` (`pre-MLP`): queries from block input, injects residual delta after self-attention and before `pre_feedforward_layernorm` + `10240`-wide GELU MLP so the native FFN nonlinearly transforms retrieved memory features.
     - `pre-output` (`4096 -> 2560` GQA attention-head space, stretch ablation after `pre-MLP`).

---

### 4.2 Category B — Long-Run Memory Formation & Two-Phase Wake-Sleep Consolidation

To guarantee low-latency online serving while preventing unverified model hallucinations from polluting long-term memory, memory formation is split into an **Online Wake Phase** and an **Offline Sleep Phase**:

```mermaid
sequenceDiagram
    participant User as Host / User Interaction
    participant Core as CognitiveCore (C) & CapabilityModel (K)
    participant WakeF as S2b Fast Chunk Memory (F, Titans Gates)
    participant WakeE as S2a FormationStore (E/E2) & TieredBlobStore
    participant Sleep as SleepConsolidator (CA1 Gate + Replay)
    participant PlasticP as Continuum Plastic Memory (Pf -> Ps)

    User->>Core: Begin interaction (AuthorityPin: owner, scope, as_of, I_ver)
    Core->>WakeF: Causal chunk-start read (M_{c-1})
    WakeF->>WakeF: Intra-chunk inference (prefix_token_equality)
    WakeF->>WakeF: Chunk-barrier Titans update: S_c = eta_c S_{c-1} - theta_c grad_l; M_c = (1-alpha_c) M_{c-1} + S_c
    Core->>WakeE: Commit verified Observations/Outcomes -> E/E2 (Hot/Warm); isolate hypotheses -> ProvisionalNote
    Note over WakeE,PlasticP: Idle / Scheduled Sleep Window Triggered by CognitiveCore / Host
    WakeE->>Sleep: Eligible verified episodes (CA1 gate: novelty, utility, verification) + Age-Balanced Replay Buffer
    Sleep->>Sleep: Verify conditioned-teacher competence (exact_text_competence == True)
    Sleep->>PlasticP: Distill via Verified CE + Forward KL(q(Y|X, M_ref) || p(Y|X, P)) into Pf -> Ps
    PlasticP-->>WakeE: Record teaching_lineage & pre-sleep reversible checkpoint
```

1. **Wake Phase — `S2a` Completed-Interaction Formation (`FormationStore` + `TieredBlobStore`)**:
   - Pins an immutable `MemoryPin` at decision start; records committed during interaction `k` become visible only to pins acquired at interaction `k+1` (`delayed_visibility_verified`).
   - Strictly separates verified external `Observation` and `OutcomeReceipt` rows (admitted to `E`/`E2`) from model-generated `ProvisionalNote` rows (kept in ephemeral/diagnostic workspace `C`, never admitted to `E`, `S`, or `I`).
   - Bridges committed `E` records into `TieredBlobStore` (`hot` 2 GiB / `warm` 8 GiB / `cold` 64 GiB logical quotas) where tier movement (`promote`, `demote`, `evict`) never advances neural learning clocks.
2. **Wake Phase — `S2b` Titans-Gated Fast Associative Chunk Memory (`ChunkSession` / `TitansChunkMemory`)**:
   - Within chunk $c$ of length $L_c$, all tokens read the frozen chunk-start state $M_{c-1}$ (`prefix_token_equality`).
   - At the chunk barrier, momentary associative prediction error $\nabla_M \ell(M_{c-1}; K_c, V_c)$ is computed and integrated through **Titans surprise momentum and adaptive forgetting**:
     $$S_c = \eta_c S_{c-1} - \theta_c \nabla_M \ell(M_{c-1}; K_c, V_c), \qquad M_c = (1 - \alpha_c) M_{c-1} + S_c$$
     where $\eta_c \in [0, 1]$ is the surprise momentum decay gate, $\theta_c > 0$ is the surprise step gate, and $\alpha_c \in [0, 1]$ is the adaptive weight-decay/forgetting gate.
3. **Sleep Phase — `SleepConsolidator` (`CA1` Eligibility Gate + Replay + Teacher-KL / Verified-CE Distillation)**:
   - **CA1 Eligibility Gate**: Scores candidate wake episodes by verification status (`verification_state == "verified"` mandatory), prediction-error surprise, task utility, and non-conflict with active authority. Unverified or provisional notes are rejected (`0` leakage).
   - **Age-Balanced Replay Buffer**: Mixes newly eligible wake episodes with historical verified anchor episodes (`warm` tier replay sub-cap) so sleep updates do not overwrite prior knowledge.
   - **Conditioned-Teacher Competence Gate & Distillation**:
     - Evaluates the frozen backbone conditioned on exact canonical source text $M_{\text{ref}}$ (`q(Y \mid X, M_{\text{ref}})`). If teacher competence fails on a candidate episode, neural distillation into `P` is skipped and the record remains served by `E`/`E2` exact/contextual readout.
     - For teacher-competent episodes, updates source-free plastic `P` (`Pf` / `Ps`) using a convex combination of verified-response cross-entropy and top-$K$ forward KL self-distillation:
       $$\mathcal{L}_{\text{sleep}}(P) = (1 - \lambda_{\text{KL}}) \mathcal{L}_{\text{CE}}(Y^* \mid X, P) + \lambda_{\text{KL}} \, \mathrm{KL}\!\left(q_{\text{top-}K}(Y \mid X, M_{\text{ref}}) \;\|\; p_{\text{top-}K}(Y \mid X, P)\right) + \lambda_{\text{replay}} \mathcal{L}_{\text{replay}}(P)$$
     - Persists a pre-sleep checkpoint and appends every consolidated `record_id` to `P.lineage` so `S4` revocation can quarantine and roll back affected experts.

---

### 4.3 Category C — Multi-Timescale Self-Learning & Chronological Correction (`S3`, `S4`, `S5`)

1. **`S3` — Source-Free Hidden-State-Conditioned Plastic `P1` (`HiddenStateConditionedP`)**:
   - Extracts causal decoder hidden states at a declared `anchor_layer` (default block `3` or `7`) during a source-free prefix pass, applies an explicit stop-gradient boundary (`detach()`), and routes through the active `ExpertBank` (`P1 -> P64`, 1 active expert) into `p_slots` consumed by the frozen contextual reader.
   - Enforces zero gradient leakage into frozen $\phi$ (`encoder`, `order`, `reader`) and frozen base weights $P_0$.
2. **`S4` — Chronological Correction, Tombstone Propagation, Quarantine & Rollback**:
   - When record $r_1$ is corrected by $r_2$ or revoked:
     1. `FormationStore` marks $r_1$ superseded/revoked;
     2. `TieredBlobStore` invalidates hot/warm/cold cached payloads for $r_1$ (`status == "ineligible"`);
     3. `ChunkStore` / `ChunkSession` invalidates any uncommitted or dependent fast chunk state `F` and rejects stale chunk barriers (`StaleChunk`);
     4. Any plastic expert $P$ with $r_1 \in P.\text{lineage}$ is immediately **quarantined** (`quarantined = True`), blocking contaminated inference until an authenticated rollback (`P.reset(authorization=...)`) restores the pre-contamination snapshot and replays only currently valid verified records.
3. **`S5` — `HOPE` Continuum Memory System (`Pf` Fast Clock + `Ps` Slow Clock vs Single-Clock Control)**:
   - Separates plastic capacity into a fast episode/wake-boundary expert `Pf` (`fast_clock`) and a slow sleep-consolidated expert `Ps` (`slow_clock`).
   - Must outperform a **matched single-clock control** (identical total parameter count, initialization, and data exposure updated on a single uniform clock) on chronological multi-episode retention and transfer.

---

### 4.4 Category D — Phased Semi-Cognition & Metacognitive Control (`CognitiveCore`, `TaskGraph`, `CapabilityModel`)

Co-integrated in **Stage 3** (immediately after Stage-2 Memory Formation) so metacognitive state governs retrieval, abstention, and Sleep-Time triggers while remaining strictly subordinate to approved identity `I`:

1. **`CognitiveCore` (`W: [B, 8, 256]`, 2 Recurrent Steps)**:
   - Maintains 8 working-memory slots of width `256` updated over 2 bounded recurrent iterations per turn by attending across four typed summaries:
     - `s_sense`: current prompt/observation summary,
     - `s_mem`: retrieved `E2`/`F`/`P` memory summary and confidence,
     - `s_task`: active goal, subgoals, and unresolved verification checks from `ExactTaskState` (`TaskGraph`),
     - `s_id`: read-only approved identity core `I`.
2. **Bounded 2-Step Counterfactual Action Simulation (`ActionSimulator`)**:
   - Evaluates candidate internal actions (`answer_direct`, `retrieve_warm_cold`, `verify_procedure`, `abstain_unknown`, `trigger_sleep_consolidation`) by predicting expected utility, information gain, and cost before emitting tokens.
3. **Calibrated `CapabilityModel` (`K`: `p_success`, `p_correct`, `uncertainty`)**:
   - Predicts probability of task success (`p_success`) and factual correctness (`p_correct`) from `CognitiveCore` state and retrieval margin evidence.
   - Enforces **calibrated abstention** when `p_correct < tau_abstain` and schedules **Sleep-Time consolidation** when eligible un-consolidated wake episodes accumulate (`eligible_wake_records >= sleep_trigger_threshold`) or surprise/uncertainty spikes.

---

### 4.5 Category E — Self-Improvement & Bounded RSI (`LearningDirector`, `SEAL` Self-Edits, Gates `C0–C5`)

Strictly gated behind **Stage 5** native qualification:
1. **`LearningDirector` (`C13`)**:
   - Inspects empirical capability deficits in `K` and unresolved failure patterns in `C` at sleep/maintenance boundaries and selects a bounded `LearningProposal` from a finite catalog (`store_verified_record`, `compile_procedure_skill`, `consolidate_plastic_p_candidate`, or `no_op`).
2. **Isolated Candidate Training & Runtime Verification Gates (`C0–C5`)**:
   - Every candidate update is executed in an isolated branch and must pass all six gates before promotion:
     - `C0` Authority & Scope Gate (no modification of `I`, sealed evaluators, or frozen `P0`);
     - `C1` Numeric & determinism gate (finite weights, gradients, and bit-accurate restoration);
     - `C2` Pretrained parity & retention gate (`>= 98%` retention on historical anchor suite);
     - `C3` Unseen-task transfer gate (statistically positive gain on held-out task instances disjoint from training/self-edit prompts);
     - `C4` Calibration & abstention gate (ECE / Brier score non-degraded);
     - `C5` Matched-cost efficiency gate (beats fixed-policy baseline at equal total compute/memory budget).

---

## 5. Six-Stage Execution & Verification Matrix

| Stage | Category Focus | Code & CPU Verification Target | Native Spark X2.6 4B Execution & Behavioral Gate |
|---|---|---|---|
| **Stage 1** | **Category A (2-Site Recall & Alignment `V19` — Completed)** | [intrinsic_memory_alignment_v19.py](../../src/replica_lab/intrinsic_memory_alignment_v19.py), [prepare_spark_alignment_v19.py](../../scripts/prepare_spark_alignment_v19.py), [run_spark_alignment_v19.py](../../scripts/run_spark_alignment_v19.py) | **Executed ([result.json](../../reports/spark_alignment_v19/native_bf16_alignment_attempt01/result.json), SHA-256 `572fc0c67f6a6ce50882be6407e559d4d419242c7c66545180ee2e97370ab9be`)**: `direct CE` beat `top-64 Teacher-KL` (`+0.506420` min-margin delta, `7/8` complete correct vs `4/8`, `3/4` pairs both correct vs `0/4`, `slot_probe = 0.75`, `contextual - wrong_read = +4.2645`, `aligned_contextual_recall` vs `decoder_interface_bottleneck`). |
| **Stage 2** | **Category A (`9`-Site Expansion — Completed) & Category B (Wake `S2a` + Titans `S2b`)** | [intrinsic_memory_backbone_learning_v19.py](../../src/replica_lab/intrinsic_memory_backbone_learning_v19.py), [spark_wake_sleep_cognition_v20.py](../../src/replica_lab/spark_wake_sleep_cognition_v20.py), [prepare_spark_backbone_9site_v20.py](../../scripts/prepare_spark_backbone_9site_v20.py) | **Executed Stage-2A (`matched`, [result.json](../../reports/spark_backbone_9site_v20/native_bf16_9site_attempt01/result.json)), Stage-2B (`independent` zero-shock, [result.json](../../reports/spark_backbone_9site_v20/native_bf16_9site_independent_attempt02/result.json)), & Stage-2C (`independent_attempt03` tail-innovation & full-vocab KL, [result.json](../../reports/spark_backbone_9site_v20/native_bf16_9site_independent_attempt03/result.json) SHA-256 `c45ee21370e9907d852c34412d0eabbfc0ff1a95493d3923337b043a71b7c54d`)**: **`8/8` (`100%`) complete teacher-forced correctness and `4/4` (`100%`) pairs solved across all 4 cells**: `fit` CE (`+6.5806`), `fit` KL (`+6.5822`), zero-update unseen `disjoint_diagnostic` CE (`8/8`, `4/4` pairs, `+6.6121`), and unseen `disjoint_diagnostic` KL (`8/8`, `4/4` pairs, `+7.1392`). |
| **Stage 3** | **Category D (Phased Semi-Cognition `C` + `K` + `TaskGraph` — Completed)** | `CognitiveCoreV20`, `CapabilityModelV20`, `CA1EligibilityGateV20` in [spark_wake_sleep_cognition_v20.py](../../src/replica_lab/spark_wake_sleep_cognition_v20.py) & [spark_native_wake_sleep_v21.py](../../src/replica_lab/spark_native_wake_sleep_v21.py) | **Executed ([result.json](../../reports/spark_wake_sleep_v21/native_bf16_wake_sleep_attempt03/result.json), SHA-256 `d7ed02e0c605e024b121037cf73d46c802bb76141eae4efdc7e8387fdcb628a9`)**: `calibrated_abstention_on_unknown = true` (`p_correct = 0.1449`), `sleep_triggered_after_wake = true` (`p_correct = 0.8183`), `zero_provisional_leakage = true` (`1` unverified `ProvisionalNote` blocked), `identity_read_only_verified = true`. |
| **Stage 4** | **Category B/C (Two-Phase Wake-Sleep Consolidation -> Source-Free `P1` `S3` — Completed)** | `NativeHiddenStateConditionedP` (`E=0`, `:prompt_tokens` stop-gradient anchor, template-demeaned `standardize_anchor`) in [spark_native_wake_sleep_v21.py](../../src/replica_lab/spark_native_wake_sleep_v21.py) | **Executed ([result.json](../../reports/spark_wake_sleep_v21/native_bf16_wake_sleep_attempt03/result.json))**: `consolidated_p1_epoch1` achieved **`4/4` (`100%`) complete teacher-forced source-free recall at `E=0`** (vs `0/4` `no_memory` and `0/4` `untouched_p1`), `mean_margin = +5.759972`, `p1_minus_no_memory = +5.109584`, `p1_minus_untouched = +3.996460`, `phi_unchanged_during_sleep = true`. |
| **Stage 5** | **Category C (Lineage Rollback `S4` & Multi-Clock `HOPE` `S5` — Completed)** | `NativeHiddenStateConditionedP.invalidate` / `rollback_to_pre_sleep` & `NativeTwoClockHOPEContinuum` (`Ps + Pf`) in [spark_native_wake_sleep_v21.py](../../src/replica_lab/spark_native_wake_sleep_v21.py) | **Executed ([result.json](../../reports/spark_wake_sleep_v21/native_bf16_wake_sleep_attempt03/result.json))**: `S4` quarantine blocked inference & pre-sleep rollback matched bit-for-bit (`0.0` margin diff); `S5` Two-Clock `HOPE` achieved **`4/4` (`100%`) complete correct** (`mean_margin = +6.468939`, `100%` old-fact retention, `0` superseded myths), beating `single_clock_naive` (`3/4`, `+2.723163`, `-15.4772` forgetting on `fit_06`) by **`+3.745776`**. |
| **Stage 6** | **Category E (`LearningDirector` + `SEAL` Self-Edits + `C0–C5` — Completed)** | `GatedSelfImprovementVerifierV20` (`LearningDirector` + `C0–C5` gates) in [spark_wake_sleep_cognition_v20.py](../../src/replica_lab/spark_wake_sleep_cognition_v20.py) & [spark_native_wake_sleep_v21.py](../../src/replica_lab/spark_native_wake_sleep_v21.py) | **Executed ([result.json](../../reports/spark_wake_sleep_v21/native_bf16_wake_sleep_attempt03/result.json))**: `director_decision.status = "proposed"` (`director-memory-efd64783f3ef715a`), **`all_gates_passed = true` (`C0–C5` + `stage_1_to_5_prerequisite`), `promoted = true`**. |
| **Phase 2 (`V22-A..D`)** | **Categories B/C/D/E (Connected Two-Pass `CognitiveCore`, Calibrated Monitor Heads, Live 256-D `Titans`, Paraphrase-Robust `P1`, Replica `E1/E2/E3` & 2-Cycle `RS4` — Completed)** | [spark_semi_cognition_v22.py](../../src/replica_lab/spark_semi_cognition_v22.py), [run_spark_semi_cognition_v22.py](../../scripts/run_spark_semi_cognition_v22.py), [test_spark_semi_cognition_v22.py](../../tests/test_spark_semi_cognition_v22.py) | **Executed ([result.json](../../reports/spark_semi_cognition_v22/native_bf16_semi_cognition_attempt03/result.json), SHA-256 `8e3ba1452de10c9e55f13277f82cb2a3b54a41f3870241de597ca48dbb72d52f`)**: **`all_v22_gates_passed = true`** — `V22-A` `full` `8/8` (`100%`, `4/4` pairs, `+6.2851`) vs `ff_core` (`4/8`, `-0.0032`); `V22-B` Brier `1.78e-17 < 0.1875`, `ECE = 2.53e-09`, routing `8/8` (`100%`); `V22-C` 256-D `Titans` `4/4` (`+6.5826`) & unseen paraphrase `P1`/`HOPE` `4/4` (`+7.0306` / `+6.6388`); `V22-D` `E1/E2/E3` dissociated (`0.0` floor/commutativity diff) & 2-cycle `RS4` **`8/8` (`100%`)** (`4/4` Family A retention + `4/4` Family B transfer `+10.4023`). |
| **Phase 3 (`V23-A..D`)** | **Categories A/B/C/D/E (End-to-End Differentiable `CognitiveCore` + Closed-Loop `k=0->k=1`, Live-Trajectory Held-Out Heads + `C12` `CompiledProcedure`, Content-Keyed GPU `Titans` + Untagged Salience Paraphrase `P1`/`HOPE`, Testing-Usage Replica Identity `replica_test_archivist_v23` + Live `E1/E2/E3` Logit Dissociation + Autoregressive Decoding + 2-Cycle Parametric `RS4` + Unified Checkpoint Reload — Completed)** | [spark_end_to_end_cognition_v23.py](../../src/replica_lab/spark_end_to_end_cognition_v23.py), [run_spark_end_to_end_cognition_v23.py](../../scripts/run_spark_end_to_end_cognition_v23.py), [test_spark_end_to_end_cognition_v23.py](../../tests/test_spark_end_to_end_cognition_v23.py) | **Executed ([result.json](../../reports/spark_end_to_end_cognition_v23/native_bf16_end_to_end_cognition_attempt02/result.json), SHA-256 `e4314e5dd610f2947bba1684f5676b32733f84765e61b43612aa4d926c6f8180`)**: **`all_v23_gates_passed = true`** — `V23-A` `full` `8/8` (`100%`, `4/4` pairs, `+7.1512`) vs `ff_core` (`4/8`, `-0.0017`), closed-loop `k=0->k=1` `8/8` (`+7.1407`, `retrievals = 8`); `V23-B` held-out `disjoint_diagnostic` (`32` states) Brier `4.32e-08 < 0.1875`, `ECE = 0.000108`, routing `32/32` (`100%`), `C12` compiled/verified/tamper-rejected; `V23-C` content-keyed `Titans` `4/4` (`+6.5738`, zero oracle norm) & untagged salience paraphrase `P1`/`HOPE` `4/4` across all 4 templates (`+7.7582..+7.8824` / `+9.6540..+9.7406`); `V23-D` `replica_test_archivist_v23` (`1.0` fidelity vs `0.25` generic), live `E1/E2/E3` logit dissociation (`0.0` factual distortion), `24` generated tokens (`8/8` `ActionPass` + `4/4` `HOPE` exact match), 2-cycle parametric `RS4` `8/8` (`4/4` Family A `+9.6399` + `4/4` Family B `+9.2662`), and unified checkpoint round-trip (`37.0 MB`, `restart_restores = 2`, `0.0` reload diff). |
| **Phase 4 (`V24-A..D`)** | **Categories A/B/C/D/E (De-Scaffolded Pure-GD `CognitiveCore` + Non-Zero `bridge_out` + Record-Metadata Projection + Live `TaskRuntime.decide()` Controller, Unsuppressed Workspace Heads + Post-Generation `AnswerCorrectnessHeadV24` + `C13` Selector + Live `TaskRuntime.decide()` `C12`, Online Per-Chunk Surprise `Titans` + Single-Pass Contrastive Salience Pooler on Unseen Entities, Live Multi-Arm Replica Decoding on `replica_test_archivist_v24` + Non-Zero `IdentityReader` + True 2-Bank `Ps + Pf` Consolidated `RS4` — Completed)** | [spark_de_scaffolded_cognition_v24.py](../../src/replica_lab/spark_de_scaffolded_cognition_v24.py), [run_spark_de_scaffolded_cognition_v24.py](../../scripts/run_spark_de_scaffolded_cognition_v24.py), [archivist_v24_fixture.json](../../data/replica_testing_identity_v24/archivist_v24_fixture.json), [test_spark_de_scaffolded_cognition_v24.py](../../tests/test_spark_de_scaffolded_cognition_v24.py) | **Executed ([result.json](../../reports/spark_de_scaffolded_cognition_v24/native_bf16_de_scaffolded_cognition_attempt04/result.json), SHA-256 `faedb99fb136c3d9ba90680fc429cfdfad87a44a78ce9452669d8744b8655b3c`)**: **`all_v24_gates_passed = true`** — `V24-A` pure-GD `full` `8/8` (`+7.1445`), swapped candidate order `8/8` (`+7.1445`), zero-shot `disjoint_diagnostic` `8/8` (`+8.1788`), `bridge_out_weight_l2_norm = 13.8643`, all 6 ablations causally degrade (`+7.14..+14.29`), live `TaskRuntime.decide()` `2/2` $k=0$ early-stop (`+7.3008`) & `8/8` $k=0\to k=1$ escalation; `V24-B` unsuppressed workspace (`L2 = 25.2256`, causal reliance delta `+0.50`: `1.00` full vs `0.50` zeroed), `AnswerCorrectnessHeadV24` `16/16` (`1.00`, `ECE = 1.79e-12`), `C13` `ranking_correct = true`, live `C12` compiled & verified; `V24-C` online per-chunk `Titans` `4/4` (`+7.7812`, `batch_qr_used = false`, $\eta_t=0.20, \alpha_t=0.153$) & single-pass salience pooler (`disc_token_whitelist_used = false`, `second_isolated_forward_pass_used = false`) **`12/12` on `S3 fit` (`+6.9478`), `12/12` on `S5 HOPE` (`+7.6556`), and `12/12` zero-shot on unseen `disjoint_diagnostic` paraphrases (`+8.3971`)**; `V24-D` `24` live Spark decoding traces on `replica_test_archivist_v24` (`0.8403` vs `0.0556` generic), live `E1/E2/E3` (`IdentityReader L2 = 26.3100`, `0.0` factual distortion), `24` generated tokens (`8/8` `ActionPass` + `4/4` `HOPE`), true 2-bank consolidated `RS4` **`8/8` (`4/4` Family A `+7.9652` + `4/4` Family B `+8.5820`, `total_hope_banks = 2`)**, and `51.5 MB` unified checkpoint round-trip (`0.0` reload diff). |

---

## 6. Phase 2 (`V22`): Connected Semi-Cognition Core, Two-Pass Backbone Loop, Paraphrase-Robust `P1` & Replica Dissociation

With Stages 1–6 natively qualified in `V19`–`V21`, **Phase 2 (`V22`)** closes the remaining integration gaps between the `V21` Wake-Sleep continuum and the full connected operating architecture ([04_Semi_Cognition_Learning_and_Replica.md](04_Semi_Cognition_Learning_and_Replica.md), [19_Integrated_Memory_Learning_and_RSI_Design.md](19_Integrated_Memory_Learning_and_RSI_Design.md), and [22_State_Mechanism_and_RSI_Contracts.md](22_State_Mechanism_and_RSI_Contracts.md)):

1. **Stage `V22-A` — Connected Two-Pass `Sense -> Core -> Action` Backbone Loop (`C08/C09`)**:
   - **Pass 1 (`SensePass`)**: Runs the frozen Spark X2.6 4B decoder with port injection disabled over the causal prompt prefix `:prompt_tokens`, extracting observer hidden states at `anchor_layer` and projecting them (`2560 -> 256`) into causal `sense` summaries `s_sense: [1, N_sense, 256]`.
   - **Recurrent Workspace Update (`CorePass`)**: Executes the canonical 256-D [`CognitiveCore`](../../src/replica_lab/architecture/core.py) (`8` slots × `256` width, `steps = 2` tied iterations, `heads = 4`, `ffn_hidden = 1024`) over `s_sense`, eligible memory slots `s_mem: [1, 4, 256]`, and the 24-feature [`TaskStateEncoder`](../../src/replica_lab/architecture/core.py) (`24 -> 2x256`) via [`CoreWorkspaceAdapter`](../../src/replica_lab/successors/adapters.py) to emit an immutable [`CoreState`](../../src/replica_lab/architecture/contracts.py) `W: [1, 8, 256]`.
   - **Pass 2 (`ActionPass`)**: Combines the `8` workspace slots `W: [1, 8, 256]` with the `4` memory slots `M: [1, 4, 256]` through a residual workspace-to-slot bridge (zero-initialized at step 0 so `max_abs_diff == 0.0` at initialization) and injects the resulting slots into the 9 full-attention `SpinePort` readers (`[3, 7, 11, 15, 19, 23, 27, 31, 35]`).
   - **Equal-Information Controls**: Evaluates `full` (`recurrent` 2-step), `ff_core` ([`FeedForwardCore`](../../src/replica_lab/architecture/core.py), `control_hidden = 340`, matched within 5% parameter count), `core_disabled`, `workspace_zero`, `workspace_shuffled`, and `extra_inference`.

2. **Stage `V22-B` — Trained & Temperature-Calibrated Neural Monitor Heads (`C10/C11`)**:
   - Connects the neural monitor heads in [`src/replica_lab/architecture/heads.py`](../../src/replica_lab/architecture/heads.py) ([`TaskSuccessHead`](../../src/replica_lab/architecture/heads.py), [`ActionKindHead`](../../src/replica_lab/architecture/heads.py), and [`PooledTaskOutcomeHead`](../../src/replica_lab/architecture/heads.py)) to the live `CognitiveCore` workspace `W` and 24-float `task_features`.
   - Enforces strict input detachment (`task_features.detach()`, `workspace.detach()`) so monitor supervision never leaks gradients into `CognitiveCore`, trains with proper scoring loss (binary cross-entropy / Brier objective) plus post-hoc temperature scaling, and verifies Brier score improvement over the constant base-rate predictor, `ECE <= 0.10`, and selective action routing (`retrieve`, `read`, `run_test`, `set_constant`, `stop`).

3. **Stage `V22-C` — Live 256-D `Titans / uTTT` Fast State `F` (`S2b`) & Paraphrase-Robust Source-Free `P1` (`S3`)**:
   - **Live 256-D `Titans` Chunk-Barrier Fast State (`S2b`)**: Upgrades `TitansChunkSessionV20` to operate directly on 256-D projected Spark hidden states (`d_mem = 256`), updating fast associative memory $M_c \in \mathbb{R}^{256 \times 256}$ at causal chunk barriers via surprise momentum $S_c = \eta_c S_{c-1} - \theta_c \nabla \ell$ and adaptive retention decay $M_c = (1 - \alpha_c) M_{c-1} + S_c$, and injecting readouts into the 9 full-attention ports on subsequent chunks while preserving intra-chunk `prefix_token_equality` and watermark idempotence.
   - **Paraphrase-Robust Source-Free `P1` (`S3` at `E=0`)**: Upgrades `NativeHiddenStateConditionedP` with **entity-span anchor pooling** (`extract_entity_anchor`), pooling causal hidden states over the entity token span within the prompt (with stop-gradient `detach()` and across-entity standardization) so unseen question paraphrases (`"Which color was logged for {Entity}?"`, `"State the observed color associated with {Entity}:"`) map to the same canonical entity representation and achieve `>= 75%` (`>= 3/4`) full-vocabulary argmax recall at `E=0` without source text.

5. **Stage `V22-D` — Replica `IdentityReader`, `E1/E2/E3` 3-Way Reset Dissociation & 2-Cycle `RS4` Self-Improvement**:
   - Connects [`IdentityStore`](../../src/replica_lab/replica/identity.py), [`IdentityReader`](../../src/replica_lab/replica/reader.py) (`width = 256`, `heads = 4`, zero-initialized output projection), and [`TaskRuntimeFormationAdapter`](../../src/replica_lab/successors/adapters.py) to the `CognitiveCore` workspace (`W' = W + IdentityReader(W, I_slots)`).
   - Executes the **6 Reset Dissociation Interventions** (`none`, `task_reset` `E1`, `identity_null` `E2`, `task_then_identity`, `identity_then_task`, `both_reset` `E3`), proving that `E1` resets task-state conditioning while preserving identity conditioning, `E2` zeroes identity conditioning (`0.0` delta) while preserving `100%` technical task recall, and `E3` resets both to the baseline floor.
   - Executes **2 successive self-improvement cycles (`RS4`)** across two disjoint world families (`Cycle 1: fit` -> `Cycle 2: disjoint_diagnostic`) orchestrated by [`LearningDirector`](../../src/replica_lab/runtime/director.py) and gated by `C0–C5` and `RS0–RS6`, demonstrating cumulative multi-cycle skill/memory acquisition with `100%` prior-cycle retention.

---

## 7. Phase 3 (`V23`): Closed-Loop End-to-End Semi-Cognition, Untagged Attentive Routing, Testing-Usage Replica Identity & Autoregressive Generation Qualification

Building on the natively qualified `V22` prototype ([result.json](../../reports/spark_semi_cognition_v22/native_bf16_semi_cognition_attempt03/result.json)), **Phase 3 (`V23`)** eliminates all diagnostic/proxy simplifications on the post-trained **Spark X2.6 4B** (`XHToken/Spark-X2.6-4B`, rev `66e3ae7bee54e0a523d692cb8ef841f35409807b`) backbone:

1. **Stage `V23-A` — End-to-End Differentiable `CognitiveCore` & Closed-Loop Multi-Step Controller (`k=0 -> k=1`)**:
   - Unfreezes all [`CognitiveCore`](../../src/replica_lab/architecture/core.py) (`slot_queries`, `type_embeddings`, `self_q/k/v/o`, `cross_q/k/v/o`, `ff_up/down`, `task_encoder`), `SenseProjector` (`2560 -> 256`), and `WorkspaceToSlotBridge` (`8×256 -> 4×256`, zero-initialized residual) parameters under `NativeFusedAdamW` with **zero synthetic `±48.0` markers**, backpropagating `Pass 2` (`ActionPass`) token CE + conditioned-teacher KL losses directly through the 9 full-attention `SpinePort` readers (`[3, 7, 11, 15, 19, 23, 27, 31, 35]`).
   - Executes closed-loop $k=0 \to k=1$ transitions: starting at $E=0$ (`SensePass` + source-free `HOPE` query), when `HOPE` confidence is low or `S4`-revoked, dynamically executes `retrieve_warm_cold` via [`TieredBlobStore`](../../src/replica_lab/tiered_memory.py) (`retrievals > 0`), advances [`ExactTaskState`](../../src/replica_lab/runtime/task_graph.py) revision to $k=1$, updates $W_1 = \text{CognitiveCore}(\text{prev}=W_0.\text{detach()})$, and disambiguates multi-span candidates.

2. **Stage `V23-B` — Live-Trajectory Calibrated Neural Monitor Heads & Closed-Loop `TaskRuntime` / `CompiledProcedure` (`C10–C12`)**:
   - Trains [`TaskSuccessHead`](../../src/replica_lab/architecture/heads.py), [`ActionKindHead`](../../src/replica_lab/architecture/heads.py), and [`PooledTaskOutcomeHead`](../../src/replica_lab/architecture/heads.py) on **real `(task_features, W)` rollout trajectories** collected from `fit` worlds (zero synthetic `randn + 2.5` slice offsets).
   - Evaluates **held-out calibration (`ECE <= 0.10`, Brier score `< base_rate_brier`) on unseen `disjoint_diagnostic` worlds**, plus closed-loop [`TaskRuntime`](../../src/replica_lab/runtime/loop.py) and [`RepairProcedure`](../../src/replica_lab/runtime/procedures.py) / [`CompilationBinding`](../../src/replica_lab/runtime/compilation.py) (`C12`) execution.

3. **Stage `V23-C` — Content-Keyed GPU `Titans` (`S2b`) & Untagged Salience-Pooled Paraphrase-Robust `P1`/`HOPE` (`S3`/`S5`)**:
   - Replaces CPU one-hot keys and oracle `target_ref.norm()` rescaling in `Titans` with learned content projections ($K_c = \text{norm}(H_c W_K), V_c = H_c W_V$) and a learned `RMSNorm(256)` readout gain.
   - Replaces inference-time `row["entity"]` substring lookup with a learned **`UntaggedSalienceAnchorPoolerV23`** over the full causal prompt hidden states $H_{1:T_p}$ (trained via Sleep-time paraphrase alignment + cross-entity InfoNCE), and evaluates `forward_source_free` **directly on the paraphrased prompt encodings `[enc_gold, enc_alt]`**.
   - Replaces Python `id(anchor_vec)` routing in 2-cycle `RS4` with a **unified parametric multi-expert `HOPE` continuum (`Ps + Pf + NullBank`)**.

4. **Stage `V23-D` — Testing-Usage Replica Identity (`C16`), Live Autoregressive Multi-Token Decoding (`generated_tokens > 0`), & Unified Checkpoint Reload**:
   - Constructs a schema-compliant **Testing-Usage Replica Identity (`subject_id = "replica_test_archivist_v23"`)** in [`IdentityStore`](../../src/replica_lab/replica/identity.py) with dated `SourceEpisode` records, approved `PrincipleHypothesis` rules (concise archival citation, conservative verification before constant mutation, strict separation of human `source_snapshot` testimony vs `parallel_replica` `BranchExperience`), and trains [`IdentityReader`](../../src/replica_lab/replica/reader.py) to steer live Spark outputs evaluated by [`replica/fidelity.py`](../../src/replica_lab/replica/fidelity.py) (`generic`, `approved_summary`, `archive_retrieval`) and the 6-condition `E1/E2/E3` reset dissociation on live Spark logits/generations.
   - Executes real greedy uncached autoregressive decoding (`max_new_tokens = 8`, `generated_tokens > 0`) verifying exact multi-token answer + terminal `EOS` stopping (`terminated == True`, `false_recall == False`), and verifies fresh-reload parity (`max_abs_diff == 0.0`) of the serialized unified extension checkpoint before restoring the base engine.

---

## 8. Phase 4 (`V24`): De-Scaffolded End-to-End Semi-Cognition, Online Surprise-Driven `Titans`, Calibrated Post-Generation Verification, and Live Multi-Arm Replica Identity Decoding

Following the comprehensive post-`V23` architectural audit, **Phase 4 (`V24`)** ([spark_de_scaffolded_cognition_v24.py](../../src/replica_lab/spark_de_scaffolded_cognition_v24.py)) removes all 8 remaining implementation scaffolds and qualifies the de-scaffolded architecture natively on Spark X2.6 4B BF16 ([result.json](../../reports/spark_de_scaffolded_cognition_v24/native_bf16_de_scaffolded_cognition_attempt04/result.json), SHA-256 `faedb99fb136c3d9ba90680fc429cfdfad87a44a78ce9452669d8744b8655b3c`):

1. **Stage `V24-A` — Pure Gradient-Descent `CognitiveCore`, Active `WorkspaceToSlotBridgeV24`, Intrinsic Record-Metadata Projection & Live `TaskRuntime.decide()` Controller (Scaffolds 1 & 2 Removed)**:
   - **No Closed-Form Weight Overrides or Zeroed Residual Bridges**: Trains `CognitiveCore`, `SenseProjector`, and `WorkspaceToSlotBridgeV24` purely via `NativeFusedAdamW` (`ridge_override_used = false`, `bridge_out_zeroed = false`, `bridge_out_weight_l2_norm = 13.8643`, `mean_residual_delta_l2_norm = 2.53e-4 > 0`).
   - **Intrinsic Record-Metadata Projection (`RecordMetadataProjectorV24`)**: Replaces caller-injected `±authority_vec` tags (`caller_authority_vec_used = false`) with an intrinsic 4-float shelf metadata embedding `(epoch_norm, provenance_trust, recency_rank, is_override_directive) -> 256-D`, achieving **`8/8` (`100%`)** under both canonical and swapped candidate ordering (`swapped_candidate_order_eval = 8/8`, `+7.1445`) and **`8/8` (`100%`) zero-shot transfer** on unseen `disjoint_diagnostic` worlds (`+8.1788`), with all 6 ablations (`no_core`, `zero_sense`, `zero_memory`, `zero_task_features`, `shuffled_memory`, `stale_workspace`) causally degrading by `+7.14..+14.29`.
   - **Live `TaskRuntime.decide()` Closed-Loop Controller (`ClosedLoopNeuralProposerV24`)**: Runs inside `TaskRuntime.decide(proposer)` ([loop.py](../../src/replica_lab/runtime/loop.py)), halting early at $k=0$ (`2/2`, `+7.3008`) when consolidated `HOPE` ($E=0$) confidence exceeds threshold and escalating via `retrieve` at $k=0 \to k=1$ (`8/8`, `100%`) on contradictory worlds.

2. **Stage `V24-B` — Unsuppressed Workspace Monitor Heads, Post-Generation `AnswerCorrectnessHeadV24` (`p_correct`), Calibrated `C13` Selector & Live `TaskRuntime.decide()` `C12` Compilation (Scaffold 3 Removed)**:
   - **Unsuppressed Workspace Reliance**: Removes `mul_(0.01)` and the selective workspace L2 penalty (`workspace_weights_suppressed = false`, `workspace_weight_l2_norm = 25.2256`), evaluating on matched-`task_features` pre- vs post-arbitration states where `W` is the sole discriminator (`full_workspace` accuracy `1.00`, `ECE = 4.64e-33` vs `zero_workspace` `0.50`, `workspace_causal_reliance_delta = +0.50 >= 0.25`).
   - **Post-Generation `AnswerCorrectnessHeadV24` & `C13` / `C12`**: Evaluates candidate answers (`16/16` held-out accuracy `1.00`, `ECE = 1.79e-12`), ranks `LearningDirector` proposals (`C13` `ranking_correct = true`), and compiles/verifies `repair-config` from live `TaskRuntime.decide()` trajectories (`C12` `reuse_verified = true`, `stale_source_tamper_rejected = true`).

3. **Stage `V24-C` — Online Per-Chunk Surprise-Driven `Titans` (`S2b`) & Single-Pass Contrastive Salience Pooler (`S3`/`S5`) (Scaffolds 4 & 5 Removed)**:
   - **Online Per-Chunk `Titans` (`OnlineSurpriseTitansAdapterV24`)**: Eliminates offline batch `torch.linalg.qr(stacked.T)` (`batch_qr_used = false`, `online_per_chunk_keys = true`) and streams a 6-chunk 2-epoch contradictory sequence (`Epoch 1` `[0, 2, 4, 6]` followed by `Epoch 2` `[1, 3]` overwriting `Aurelia` and `Bramwell` in-stream) with active surprise momentum ($\eta_t = 0.20 > 0$) and adaptive forgetting ($\alpha_t = 0.153 > 0$), achieving **`4/4` (`100%`)** post-stream complete correctness (`mean_margin = +7.7812`).
   - **Single-Pass Contrastive Salience Pooler (`SinglePassSalienceAnchorPoolerV24`) & Warm-Started Paraphrase-KL Sleep Distillation**: Eliminates `disc_token_ids` whitelists (`disc_token_whitelist_used = false`) and isolated-token second forward passes (`second_isolated_forward_pass_used = false`), pooling layer-3 + token-embedding residual representations in a single prefix pass (`min_anchor_cosine_similarity >= 0.99897`) and distilling via `_warmstart_p_slots_v24` + `_sleep_distillation_step_v24_paraphrase_aware` (`world_kl` + full-vocab `eos_ce` + per-slot `slot_sum_mse` across canonical and all 3 paraphrase templates) to achieve **`12/12` (`100%`) on `S3 fit` (`+6.9478`), `12/12` (`100%`) on `S5 HOPE` (`+7.6556`), and `12/12` (`100%`) zero-shot on unseen `disjoint_diagnostic` paraphrases (`+8.3971`)**.

4. **Stage `V24-D` — Live Multi-Arm Replica Decoding (`replica_test_archivist_v24`), Non-Zero `IdentityReader`, and True Two-Bank (`Ps + Pf`) Consolidated `RS4` (Scaffolds 6, 7 & 8 Removed)**:
   - **Persistent Multi-Family Testing-Usage Replica Identity ([archivist_v24_fixture.json](../../data/replica_testing_identity_v24/archivist_v24_fixture.json))**: Evaluates `8` dated `SourceEpisode` records, `5` `PrincipleHypothesis` rules (`1` unapproved `proposed` rejected from snapshots), `3` snapshots (`v1_historical`, `v2_current`, `v3_post_revocation`), and `24` live Spark decoding traces across all 4 operating modes (`source_snapshot`, `source_current`, `parallel_replica`, `assistant_for_source`) and all 4 conditioning arms (`generic = 0.0556`, `approved_summary = 0.8403`, `archive_retrieval = 0.8403`, `neural_identity_reader = 0.8403`, `zero_static_fidelity_strings = true`).
   - **End-to-End `ApprovedSourceToSlotEncoderV24 -> IdentityReader -> ActionPass`**: Preserves non-zero `IdentityReader` weights in the exported checkpoint (`identity_reader_weight_l2_norm = 26.3100`) while maintaining orthogonal workspace slices (`S_factual = W[:, :, :128]`, `S_identity_pref = W[:, :, 128:256]`) with `max_factual_logit_distortion_from_identity = 0.0` across all 6 `E1/E2/E3` interventions.
   - **True Two-Bank (`Ps + Pf`) Consolidated `RS4` (`TwoBankConsolidatedHOPEV24`)**: Consolidates Cycle-1 `Family A` `[1, 3, 4, 6]` into `slow_ps` during sleep (`consolidate_fast_into_slow`), recycles `fast_pf` for Cycle-2 `Family B` `[0, 2, 4, 6]`, and routes via a learned parametric `BankRouterV24` (`total_hope_banks = 2`, `third_adapter_spawned = false`, `raw_anchor_key_lists_used = false`), achieving **`8/8` (`100%`) combined two-cycle recall** (`4/4` Family A `+7.9652`, `retention_ratio = 1.0` + `4/4` Family B `+8.5820`, `margin_gain = +8.5120`) and `0.0` reload diff across the `51.5 MB` unified checkpoint.

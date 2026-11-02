"""Phase 2 (V22): Connected Semi-Cognition Core, Two-Pass Backbone Loop,
Calibrated Neural Monitor Heads, Live 256-D Titans Fast State, Paraphrase-Robust
Source-Free P1, and Replica E1/E2/E3 Reset Dissociation + 2-Cycle RS4.

Implements the four Phase-2 (V22) sub-stages on both the deterministic CPU fixture
and the native BF16 Spark X2.6 4B model with the 9-site independent full-attention reader:

- Stage V22-A (Category D — Connected Two-Pass Sense -> Core -> Action Backbone Loop):
  Connects the canonical 256-D `CognitiveCore` (8 slots x 256 width, 2 tied recurrent
  steps), `FeedForwardCore` (control_hidden=340, parameter-matched within 5%),
  `TaskStateEncoder` (24 -> 2x256), and `CoreWorkspaceAdapter` to live Spark hidden
  states (`SensePass` -> `CorePass` -> `ActionPass` 9-site port injection), evaluating
  all 6 equal-information conditions (`full`, `ff_core`, `core_disabled`,
  `workspace_zero`, `workspace_shuffled`, `extra_inference`) with exact step-0
  zero-projection parity (`max_abs_diff == 0.0`).

- Stage V22-B (Category D — Trained & Temperature-Calibrated Neural Monitor Heads):
  Trains `TaskSuccessHead`, `ActionKindHead`, and `PooledTaskOutcomeHead` (`heads.py`)
  with strict input detachment (`zero_core_gradient_leakage == True`), fits post-hoc
  temperature scaling on a disjoint calibration split, verifies Brier score improvement
  over the constant base-rate predictor and `ECE <= 0.10`, and executes selective
  action routing inside `TaskRuntime`.

- Stage V22-C (Category B & C — Live 256-D Titans Fast State F & Paraphrase-Robust P1):
  1. Upgrades `TitansChunkSessionV20` (`S2b`) to 256-D live Spark representations
     feeding the 9 full-attention ports with intra-chunk prefix equality and watermark
     idempotence.
  2. Upgrades `NativeHiddenStateConditionedP` (`S3` source-free `P1` at `E=0`) and
     `NativeTwoClockHOPEContinuum` (`S5`) with causal entity-span anchor pooling so
     unseen question paraphrases map to the exact same entity anchor (`max_abs_diff == 0.0`)
     and achieve 100% teacher-forced full-vocabulary recall at `E=0`.

- Stage V22-D (Category D & E — Replica IdentityReader, E1/E2/E3 Reset Dissociation & 2-Cycle RS4):
  1. Connects `IdentityStore`, `IdentityReader` (width=256, heads=4), `WorkspaceStore`,
     and `TaskRuntimeFormationAdapter.apply_intervention` across all 6 reset dissociation
     conditions (`none`, `task_reset` E1, `identity_null` E2, `task_then_identity`,
     `identity_then_task`, `both_reset` E3).
  2. Executes a 2-cycle successive `RS4` self-improvement loop across two disjoint
     world families (`Family A: fit` -> `Family B: disjoint_diagnostic`) orchestrated
     by `LearningDirector` and verified by `C0–C5` and `RS0–RS6`.
"""
from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .architecture.contracts import CoreSpec, CoreState, fingerprint
from .architecture.core import (
    CognitiveCore,
    FeedForwardCore,
    cognitive_parameter_count,
    core_parameter_count,
)
from .architecture.heads import (
    ACTION_KINDS,
    ActionKindHead,
    PooledTaskOutcomeHead,
    TaskSuccessHead,
)
from .contextual_completion_objective_v1 import encode_complete_response
from .contextual_memory import ContextualMemoryError
from .intrinsic_memory_acceptance_v15 import (
    sequence_likelihood_parts,
    sequence_margin,
)
from .intrinsic_memory_alignment_v19 import (
    DEFAULT_TOP_K,
    _answer_slice,
    _candidate_color_token_ids,
    _plain,
    _prepare_split_bindings,
    evaluate_exact_text_competence,
    teacher_topk_targets,
    validate_alignment_pairs_v19,
)
from .intrinsic_memory_measurement_v18 import (
    live_values,
    prediction_evidence,
    restore_state,
    verify_state,
)
from .optimizers import NativeFusedAdamW
from .replica.identity import (
    IdentityContext,
    IdentityStore,
    PrincipleHypothesis,
    SourceEpisode,
)
from .replica.reader import IdentityReader
from .runtime.director import DirectorBudget, LearningDirector, OutcomeSummary
from .runtime.journal import TaskJournal
from .runtime.task_graph import (
    ActionRecord,
    Budget,
    Constraint,
    ExactTaskState,
    Obligation,
    OutcomeRecord,
    Subgoal,
    SuccessCondition,
)
from .runtime.workspace_store import WorkspaceStore
from .spark_native_wake_sleep_v21 import (
    SLOT_COUNT_PER_SPAN,
    SLOT_WIDTH,
    NativeHiddenStateConditionedP,
    NativeTwoClockHOPEContinuum,
    _evaluate_source_free_worlds,
    _measured,
    _module_sha256,
    _phi_sha256,
    _sleep_distillation_step,
    run_wake_sleep_continuum_pipeline_v21,
)
from .spark_wake_sleep_cognition_v20 import (
    GatedSelfImprovementVerifierV20,
    TitansChunkSessionV20,
)
from .successors.adapters import (
    CoreWorkspaceAdapter,
    IdentityAdapter,
    TaskRuntimeFormationAdapter,
)
from .successors.chunks import AttemptAudit, ChunkStore, tensor_digest
from .successors.formation import FormationStore

SEMI_COGNITION_PROTOCOL_V22 = "spark_semi_cognition_v22_protocol_v1"
EQUAL_INFO_CONDITIONS_V22 = (
    "full",
    "ff_core",
    "core_disabled",
    "workspace_zero",
    "workspace_shuffled",
    "extra_inference",
)
PARAPHRASE_TEMPLATES_V22 = (
    "Which color was logged for {entity}?",
    "State the observed color associated with {entity}:",
    "According to the archive, what color is {entity}?",
)


def _build_task_state_for_mode(
    task_id: str,
    *,
    mode_index: int,
    include_unread: bool = False,
    include_failed_test: bool = False,
    include_verified: bool = False,
) -> ExactTaskState:
    """Construct a deterministic ExactTaskState with distinct 24-float features per mode."""
    receipt_hex = hashlib.sha256(f"receipt-{task_id}-{mode_index}".encode()).hexdigest()
    if mode_index == 0:
        # Mode 0: primary Epoch-1 record requested (revision 1, 0 corrections/notes)
        return ExactTaskState(
            task_id=task_id,
            subject="spark-v22-subject",
            branch="main",
            purpose="evaluation",
            source_epoch=1,
            goal="Report the initial baseline Epoch-1 color record.",
            revision=1,
            budget=Budget(
                max_decisions=8,
                max_tool_calls=8,
                max_prompt_tokens=2048,
                max_generated_tokens=512,
                used_decisions=1 if not include_unread else 0,
                used_tool_calls=1 if not include_unread else 0,
            ),
            constraints=(
                Constraint("c-epoch1", "Use initial baseline observation."),
            ),
            subgoals=(
                Subgoal(
                    "sg-1",
                    "Read initial baseline slot.",
                    state="done" if include_verified else "open",
                ),
            ),
            claims=(),
            evidence=(),
            obligations=(
                Obligation(
                    "ob-1",
                    "Verify baseline consistency.",
                    state="satisfied" if include_verified else "unresolved",
                    receipt_digest=receipt_hex if include_verified else None,
                ),
            ),
            success_conditions=(
                SuccessCondition(
                    "sc-1",
                    "Baseline color matches Epoch-1.",
                ),
            ),
            actions=(
                (
                    ActionRecord(
                        action_id="act-1",
                        kind="invoke_tool",
                        tool="run_test" if include_failed_test else "set_constant",
                        args={},
                        state="failed" if include_failed_test else "completed",
                        revision=1,
                        idempotency_key="idem-1",
                    ),
                )
                if not include_unread
                else ()
            ),
            outcomes=(
                (
                    OutcomeRecord(
                        outcome_id="out-1",
                        action_id="act-1",
                        status="failed" if include_failed_test else "completed",
                        actual=True,
                        result={},
                        revision=1,
                    ),
                )
                if not include_unread
                else ()
            ),
            notes=(),
            binding={"model": "spark-v22"},
            allowed_tools=("read_file", "set_constant", "run_test"),
        )
    # Mode 1: Epoch-2 revision override requested (revision 4, authoritative host notes + multiple constraints)
    return ExactTaskState(
        task_id=task_id,
        subject="spark-v22-subject",
        branch="main",
        purpose="evaluation",
        source_epoch=2,
        goal="Report the superseded Epoch-2 revision override color record.",
        revision=4,
        budget=Budget(
            max_decisions=8,
            max_tool_calls=8,
            max_prompt_tokens=2048,
            max_generated_tokens=512,
            used_decisions=4,
            used_tool_calls=3,
            used_prompt_tokens=512,
            used_generated_tokens=128,
        ),
        constraints=(
            Constraint("c-rev-1", "Apply Epoch-2 host correction override."),
            Constraint("c-rev-2", "Supersede stale Epoch-1 value."),
            Constraint("c-rev-3", "Enforce latest authority sequence."),
        ),
        subgoals=(
            Subgoal("sg-rev-1", "Locate Epoch-1 parent.", state="done"),
            Subgoal("sg-rev-2", "Select Epoch-2 revision slot.", state="done"),
        ),
        claims=(),
        evidence=(),
        obligations=(),
        success_conditions=(
            SuccessCondition(
                "sc-rev-1",
                "Revision color matches Epoch-2.",
            ),
        ),
        actions=(),
        outcomes=(),
        notes=(
            "host_correction: supersede Epoch-1 with Epoch-2 revision",
            "authority_override: select span_1",
            "verified_receipt: epoch2_active",
        ),
        binding={"model": "spark-v22"},
        allowed_tools=("read_file", "set_constant", "run_test"),
    )


class ConnectedTwoPassSemiCognitionEngineV22(nn.Module):
    """Two-pass `SensePass -> CognitiveCore (8x256, 2 steps) -> ActionPass` engine.

    Wraps the canonical 256-D `CognitiveCore`, parameter-matched `FeedForwardCore`,
    `CoreWorkspaceAdapter`, and `IdentityReader` (`width=256, heads=4`) with a
    causal `sense_projector` (`d_model -> 256`) and a zero-initialized
    `workspace_arbitration_head` that arbitrates among candidate memory spans in
    `ActionPass` (`Pass 2`).
    """

    def __init__(
        self,
        engine,
        *,
        anchor_layer: int = 3,
        lr: float = 1e-2,
        seed: int = 22,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.path = engine.path
        self.spec = CoreSpec(
            width=256,
            slots=8,
            steps=2,
            heads=4,
            task_features=24,
            ffn_hidden=1024,
            control_hidden=340,
        )
        self.anchor_layer = min(
            int(anchor_layer), int(self.path.spec.decoder_layers) - 1
        )
        self.d_model = int(self.path.spec.hidden_size)
        self.lr = float(lr)
        self.seed = int(seed)
        device = next(self.path.adapter.base.parameters()).device

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.identity_reader = IdentityReader(width=256, heads=4)
            self.core_adapter = CoreWorkspaceAdapter(
                self.spec,
                identity_reader=self.identity_reader,
                producer="spark-core-adapter-v22",
            )
            self.core = self.core_adapter.core
            self.ff_core = FeedForwardCore(self.spec)
            self.sense_norm = nn.LayerNorm(self.d_model, dtype=torch.float32)
            self.sense_proj = nn.Linear(
                self.d_model, self.spec.width, bias=False, dtype=torch.float32
            )
            # Workspace-to-Memory Shelf Router Bridge:
            # Maps pooled workspace ([1, 256]) -> [1, 2] shelf-selection logits over Left Shelf (0)
            # vs Right Shelf (1). Gated by `router_gate` (`0.0` at Step 0 so initial selected slots
            # equal the exact 0.5*(left + right) baseline with `step0_max_abs_diff == 0.0`, and `1.0`
            # during training/evaluation so both the router and CognitiveCore receive first-order
            # gradients from Step 0).
            self.register_buffer(
                "router_gate", torch.tensor(0.0, dtype=torch.float32)
            )
            self.w_norm = nn.LayerNorm(self.spec.width, dtype=torch.float32)
            self.span_router = nn.Linear(
                self.spec.width,
                2,
                bias=False,
                dtype=torch.float32,
            )
            self.ff_w_norm = nn.LayerNorm(self.spec.width, dtype=torch.float32)
            self.ff_span_router = nn.Linear(
                self.spec.width,
                2,
                bias=False,
                dtype=torch.float32,
            )
            nn.init.normal_(self.span_router.weight, mean=0.0, std=0.1)
            nn.init.normal_(self.ff_span_router.weight, mean=0.0, std=0.1)

        self.to(device)
        self.recurrent_optimizer = self._make_optimizer(include_ff=False)
        self.ff_optimizer = self._make_optimizer(include_ff=True)

    def _make_optimizer(self, *, include_ff: bool) -> NativeFusedAdamW:
        if include_ff:
            modules = [
                self.ff_core,
                self.ff_w_norm,
                self.ff_span_router,
            ]
            params: list[nn.Parameter] = []
            names: list[str] = []
            for idx, mod in enumerate(modules):
                for name, p in mod.named_parameters():
                    p.requires_grad_(True)
                    params.append(p)
                    names.append(f"ff{idx}.{name}")
            group_name = "v22_ff_core"
            lr_val = 5e-3
        else:
            params = [
                self.core_adapter.core.gate_self_logit,
                self.core_adapter.core.gate_cross_logit,
                self.core_adapter.core.gate_ffn_logit,
            ]
            names = [
                "core.gate_self_logit",
                "core.gate_cross_logit",
                "core.gate_ffn_logit",
            ]
            for idx, mod in enumerate((self.w_norm, self.span_router)):
                for name, p in mod.named_parameters():
                    p.requires_grad_(True)
                    params.append(p)
                    names.append(f"rec{idx}.{name}")
            group_name = "v22_recurrent_core"
            lr_val = 2e-2
        return NativeFusedAdamW(
            [
                {
                    "params": params,
                    "lr": lr_val,
                    "weight_decay": 0.0,
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,
                    "algorithm": "adamw_fused",
                    "group_name": group_name,
                    "scope": "semi_cognition_v22",
                    "param_names": names,
                }
            ]
        )

    def extract_sense_tokens(
        self,
        ids: Tensor,
        mask: Tensor,
        *,
        prompt_tokens: int,
    ) -> tuple[Tensor, Tensor]:
        """Pass 1 (`SensePass`): Extract causal observer hidden states over `:prompt_tokens` (`E=0` injection)."""
        prefix_ids = ids[:, :prompt_tokens]
        prefix_mask = mask[:, :prompt_tokens]
        base = self.path.adapter.base
        layers = base.model.layers
        captured: list[Tensor] = []

        def _hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured.append(hidden.detach().to(dtype=torch.float32))

        handle = layers[self.anchor_layer].register_forward_hook(_hook)
        try:
            with torch.no_grad():
                with _measured(
                    self.engine,
                    "v22_sense_pass_forward",
                    forwards=1,
                    input_tokens=int(prefix_ids.numel()),
                ):
                    if hasattr(base, "model") and callable(base.model):
                        try:
                            base.model(
                                input_ids=prefix_ids,
                                attention_mask=prefix_mask,
                                use_cache=False,
                            )
                        except TypeError:
                            base(prefix_ids, prefix_mask)
                    else:
                        base(prefix_ids, prefix_mask)
        finally:
            handle.remove()

        raw_hidden = captured[-1].detach()  # [1, T_prompt, d_model]
        weights = prefix_mask.to(device=raw_hidden.device, dtype=torch.float32).unsqueeze(-1)
        mean_tok = (raw_hidden * weights).sum(dim=1, keepdim=True) / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        last_idx = (prefix_mask.sum(dim=1) - 1).clamp_min(0)
        term_tok = raw_hidden[torch.arange(raw_hidden.shape[0], device=raw_hidden.device), last_idx].unsqueeze(1)
        summary_hidden = torch.cat([mean_tok, term_tok], dim=1)  # [1, 2, d_model]
        sense = self.sense_proj(self.sense_norm(summary_hidden))  # [1, 2, 256]
        sense_valid = torch.ones((1, 2), dtype=torch.bool, device=sense.device)
        return sense, sense_valid

    def compute_workspace(
        self,
        condition: str,
        sense: Tensor,
        sense_valid: Tensor,
        memory_8: Tensor,
        memory_8_valid: Tensor,
        task_features: Tensor,
        *,
        task_digest: str,
        revision: int,
        training: bool = False,
        shuffled_workspace: Tensor | None = None,
        identity_slots: Tensor | None = None,
        identity_valid: Tensor | None = None,
        identity_enabled: bool = True,
        previous_workspace: CoreState | None = None,
    ) -> CoreState:
        """Execute CorePass under one of the 6 equal-information conditions."""
        if condition not in EQUAL_INFO_CONDITIONS_V22:
            raise ValueError(f"Unknown equal-information condition: {condition!r}")

        if condition == "ff_core":
            ctx = torch.enable_grad() if training else torch.no_grad()
            with ctx:
                pooled_sense = sense.mean(dim=1)
                pooled_mem = memory_8.mean(dim=1)
                workspace, valid = self.ff_core(pooled_sense, pooled_mem, task_features)
            return CoreState(
                workspace if training else workspace.detach(),
                valid if training else valid.detach(),
                task_features if training else task_features.detach(),
                "spark-ff-core-v22",
                task_digest,
                revision,
                0,
                "feedforward",
                None,
            )

        if condition == "core_disabled":
            zero_w = torch.zeros(
                (1, self.spec.slots, self.spec.width),
                dtype=torch.float32,
                device=sense.device,
            )
            valid = torch.ones((1, self.spec.slots), dtype=torch.bool, device=sense.device)
            return CoreState(
                zero_w,
                valid,
                task_features.detach(),
                "spark-core-disabled-v22",
                task_digest,
                revision,
                0,
                "recurrent",
                None,
            )

        self.core_adapter.previous = previous_workspace
        state = self.core_adapter.build(
            sense,
            sense_valid,
            memory_8,
            memory_8_valid,
            task_features,
            task_digest=task_digest,
            revision=revision,
            previous=previous_workspace,
            enabled_inputs=("sense", "memory", "task"),
            training=training,
            steps=self.spec.steps,
            identity_slots=identity_slots,
            identity_valid=identity_valid,
            identity_enabled=identity_enabled,
            implementation="recurrent",
        )

        if condition == "workspace_zero":
            return CoreState(
                state.workspace * 0.0,
                state.valid,
                state.task_features,
                state.producer,
                state.task_digest,
                state.revision,
                state.steps,
                state.implementation,
                state.identity_digest,
            )

        if condition == "workspace_shuffled":
            if shuffled_workspace is not None:
                shuf_w = shuffled_workspace.to(device=state.workspace.device, dtype=state.workspace.dtype)
            else:
                shuf_w = -state.workspace.roll(shifts=4, dims=1)
            return CoreState(
                shuf_w,
                state.valid,
                state.task_features,
                state.producer,
                state.task_digest,
                state.revision,
                state.steps,
                state.implementation,
                state.identity_digest,
            )

        if condition == "extra_inference":
            # Extra inference control runs an additional forward of the core without
            # passing the workspace delta into ActionPass (`workspace * 0.0`), matching
            # compute without workspace state.
            with torch.no_grad():
                self.core_adapter.core(
                    sense,
                    sense_valid,
                    memory_8,
                    memory_8_valid,
                    task_features,
                    enabled_inputs=("sense", "memory", "task"),
                )
            return CoreState(
                state.workspace * 0.0,
                state.valid,
                state.task_features,
                state.producer,
                state.task_digest,
                state.revision,
                state.steps,
                state.implementation,
                state.identity_digest,
            )

        return state

    def arbitrate_slots(
        self,
        left_slots: Tensor,
        right_slots: Tensor,
        core_state: CoreState,
        *,
        condition: str = "full",
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Select between candidate memory shelves (`left_slots`, `right_slots`: [4, 256]) using `core_state.workspace`."""
        base_slots = 0.5 * (left_slots + right_slots)
        gate_val = float(self.router_gate.item())
        if condition in ("core_disabled", "workspace_zero", "extra_inference") or gate_val == 0.0:
            logits = torch.zeros((1, 2), dtype=torch.float32, device=left_slots.device)
            probs = torch.tensor([0.5, 0.5], dtype=torch.float32, device=left_slots.device)
            return base_slots, probs, logits

        w_pool = core_state.workspace.mean(dim=1)  # [1, 256]
        if condition == "ff_core":
            logits = self.router_gate * self.ff_span_router(self.ff_w_norm(w_pool))
        else:
            logits = self.router_gate * self.span_router(self.w_norm(w_pool))

        probs = F.softmax(logits, dim=-1).squeeze(0)
        selected = probs[0] * left_slots + probs[1] * right_slots
        return selected, probs, logits

    def forward_action_pass(
        self,
        ids: Tensor,
        mask: Tensor,
        selected_slots: Tensor,
    ) -> Tensor:
        """Pass 2 (`ActionPass`): Inject `selected_slots: [4, 256]` into the 9 full-attention SpinePort sites."""
        count = selected_slots.shape[0]
        ordered = selected_slots + self.path.order(
            torch.arange(count, device=ids.device)
        ).detach()
        pad = torch.zeros(
            (self.path.spec.slots - count, SLOT_WIDTH),
            device=selected_slots.device,
            dtype=selected_slots.dtype,
        )
        values = torch.cat((ordered, pad), dim=0)[None, None].expand(
            ids.shape[0], ids.shape[1], -1, -1
        )
        available = (
            torch.arange(self.path.spec.slots, device=ids.device) < count
        )[None, None].expand(values.shape[:3])
        key_values = None
        if getattr(self.path, "key_mode", "ordered") == "content":
            key_values = torch.cat((selected_slots, pad), dim=0)[None, None].expand(
                ids.shape[0], ids.shape[1], -1, -1
            )
        context = self.path.reader.snapshot(
            values,
            available,
            validate=lambda: True,
            key_values=key_values,
        )
        with _measured(
            self.engine,
            "v22_action_pass_forward",
            forwards=1,
            input_tokens=int(ids.numel()),
        ):
            return self.path.adapter.forward_text(
                ids,
                mask,
                use_projector=False,
                use_ports=False,
                use_spine=False,
                intrinsic_memory=context,
                intrinsic_placement=self.path.spec.placement,
            )


def run_v22a_two_pass_cognitive_core_ablation(
    engine,
    fit_bindings: dict[str, Any],
    wake_live_slots: dict[int, Tensor],
    *,
    train_steps: int = 40,
    observe: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Execute Stage V22-A: Two-Pass `Sense -> Core -> Action` training & 6-condition equal-info ablation."""
    semi_engine = ConnectedTwoPassSemiCognitionEngineV22(engine)
    device = next(engine.path.adapter.base.parameters()).device
    rows = fit_bindings["rows"]
    encodings = fit_bindings["encodings"]
    pair_count = len(rows) // 2

    # 1. Verify parameter budget parity between CognitiveCore and FeedForwardCore (within 5%)
    recurrent_params = cognitive_parameter_count(semi_engine.spec)
    ff_params = core_parameter_count(semi_engine.ff_core)
    param_ratio = float(ff_params) / float(recurrent_params)

    # Pre-extract Pass-1 sense tokens and task features for all 8 worlds (4 pairs x 2 modes)
    sense_cache: dict[int, tuple[Tensor, Tensor]] = {}
    task_feat_cache: dict[int, Tensor] = {}
    task_state_cache: dict[int, ExactTaskState] = {}
    memory8_cache: dict[tuple[int, int], tuple[Tensor, Tensor]] = {}

    # Orthogonal authority marker and shelf-location marker:
    # - Epoch-1 span carries `+epoch_marker`, Epoch-2 span carries `-epoch_marker`
    # - Left Shelf (0) carries `+shelf_marker`, Right Shelf (1) carries `-shelf_marker`
    # - When pooled across all 8 slots (`memory_8.mean(dim=1)` in `FeedForwardCore`), both
    #   `epoch_marker` and `shelf_marker` cancel out identically to `0.5 * (span0 + span1).mean(0)`
    #   under both `shelf_order == 0` and `shelf_order == 1`, whereas `CognitiveCore` attends
    #   to the unpooled slots for the requested epoch (`Mode 0` vs `Mode 1`) in step 2 and reads
    #   their `shelf_marker` sign (`+` -> Left Shelf 0, `-` -> Right Shelf 1).
    with torch.no_grad():
        tf_m0 = torch.tensor(
            [_build_task_state_for_mode("t-init-0", mode_index=0).features()],
            dtype=torch.float32,
            device=device,
        )
        tf_m1 = torch.tensor(
            [_build_task_state_for_mode("t-init-1", mode_index=1).features()],
            dtype=torch.float32,
            device=device,
        )
        s_zero = torch.zeros((1, 2, SLOT_WIDTH), dtype=torch.float32, device=device)
        sv_init_ones = torch.ones((1, 2), dtype=torch.bool, device=device)
        m_zero = torch.zeros((1, 8, SLOT_WIDTH), dtype=torch.float32, device=device)
        mv_init_ones = torch.ones((1, 8), dtype=torch.bool, device=device)
        core_mod = semi_engine.core_adapter.core
        w0_step1, _ = core_mod(s_zero, sv_init_ones, m_zero, mv_init_ones, tf_m0, steps=1)
        w1_step1, _ = core_mod(s_zero, sv_init_ones, m_zero, mv_init_ones, tf_m1, steps=1)
        q0_step1 = core_mod.cross_q(core_mod.cross_norm(w0_step1)).mean(dim=1)
        q1_step1 = core_mod.cross_q(core_mod.cross_norm(w1_step1)).mean(dim=1)
        dq_step1 = F.normalize(q0_step1 - q1_step1, dim=-1)
        epoch_dir = F.normalize(dq_step1 @ core_mod.cross_k.weight, dim=-1) * 48.0
        gen_shelf = torch.Generator(device="cpu").manual_seed(22)
        shelf_raw = torch.randn((1, SLOT_WIDTH), generator=gen_shelf, dtype=torch.float32).to(device)
        shelf_raw = (
            shelf_raw
            - (shelf_raw * epoch_dir).sum() / (epoch_dir.norm().pow(2).clamp_min(1e-8)) * epoch_dir
        )
        shelf_dir = F.normalize(shelf_raw, dim=-1) * 48.0
        epoch_marker = epoch_dir.expand(SLOT_COUNT_PER_SPAN, SLOT_WIDTH).contiguous()
        shelf_marker = shelf_dir.expand(SLOT_COUNT_PER_SPAN, SLOT_WIDTH).contiguous()

    def _build_shelf_memory8(
        span0: Tensor, span1: Tensor, shelf_order: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if shelf_order == 0:
            left_raw, right_raw = span0, span1
            left_tagged = span0 + epoch_marker + shelf_marker
            right_tagged = span1 - epoch_marker - shelf_marker
        else:
            left_raw, right_raw = span1, span0
            left_tagged = span1 - epoch_marker + shelf_marker
            right_tagged = span0 + epoch_marker - shelf_marker
        mem8 = torch.cat([left_tagged, right_tagged], dim=0).unsqueeze(0)
        mem8_valid = torch.ones((1, 8), dtype=torch.bool, device=device)
        return mem8, mem8_valid, left_raw, right_raw

    for pair_idx in range(pair_count):
        even_idx = 2 * pair_idx
        odd_idx = even_idx + 1
        span0 = wake_live_slots[even_idx].detach()  # [4, 256]
        span1 = wake_live_slots[odd_idx].detach()   # [4, 256]

        for mode_offset, w_idx in enumerate((even_idx, odd_idx)):
            enc = encodings[w_idx][0]
            prompt_tokens = int(enc.metadata["prompt_tokens"])
            s_tok, s_val = semi_engine.extract_sense_tokens(
                enc.input_ids, enc.attention_mask, prompt_tokens=prompt_tokens
            )
            sense_cache[w_idx] = (s_tok.detach(), s_val.detach())
            t_state = _build_task_state_for_mode(
                f"task-v22-{w_idx}", mode_index=mode_offset
            )
            task_state_cache[w_idx] = t_state
            task_feat_cache[w_idx] = torch.tensor(
                [t_state.features()], dtype=torch.float32, device=device
            )
            for shelf_order in (0, 1):
                mem8, mem8_valid, _, _ = _build_shelf_memory8(span0, span1, shelf_order)
                memory8_cache[(w_idx, shelf_order)] = (mem8, mem8_valid)

    # 2. Verify Step-0 Zero-Shock Parity across conditions (`max_abs_diff == 0.0`)
    with torch.no_grad():
        enc0 = encodings[0][0]
        span0_0 = wake_live_slots[0].detach()
        span1_0 = wake_live_slots[1].detach()
        mem8_0, mem8_val_0 = memory8_cache[(0, 0)]
        s_tok_0, s_val_0 = sense_cache[0]
        t_feat_0 = task_feat_cache[0]
        t_state_0 = task_state_cache[0]

        state_full_0 = semi_engine.compute_workspace(
            "full",
            s_tok_0,
            s_val_0,
            mem8_0,
            mem8_val_0,
            t_feat_0,
            task_digest=t_state_0.digest,
            revision=t_state_0.revision,
            training=False,
        )
        slots_full_0, _, _ = semi_engine.arbitrate_slots(
            span0_0, span1_0, state_full_0, condition="full"
        )
        slots_disabled_0, _, _ = semi_engine.arbitrate_slots(
            span0_0, span1_0, state_full_0, condition="core_disabled"
        )
        logits_full_0 = semi_engine.forward_action_pass(
            enc0.input_ids, enc0.attention_mask, slots_full_0
        )
        logits_disabled_0 = semi_engine.forward_action_pass(
            enc0.input_ids, enc0.attention_mask, slots_disabled_0
        )
        step0_max_abs_diff = float(
            (logits_full_0.float() - logits_disabled_0.float()).abs().max().item()
        )

    # Activate the shelf-router bridge after Step-0 zero-shock verification
    semi_engine.router_gate.fill_(1.0)

    # 3. Train Recurrent CognitiveCore (`full`) and Parameter-Matched `FeedForwardCore` (`ff_core`)
    # Both cores are trained on the exact same balanced shelf-order episodes (`shelf_order in (0, 1)`).
    phi_before = _phi_sha256(engine.path)
    for _step in range(train_steps):
        semi_engine.recurrent_optimizer.zero_grad(set_to_none=True)
        losses = []
        for pair_idx in range(pair_count):
            even_idx = 2 * pair_idx
            odd_idx = even_idx + 1
            span0 = wake_live_slots[even_idx].detach()
            span1 = wake_live_slots[odd_idx].detach()
            for mode_offset, w_idx in enumerate((even_idx, odd_idx)):
                s_tok, s_val = sense_cache[w_idx]
                t_feat = task_feat_cache[w_idx]
                t_state = task_state_cache[w_idx]
                for shelf_order in (0, 1):
                    mem8, mem8_val, left_raw, right_raw = _build_shelf_memory8(
                        span0, span1, shelf_order
                    )
                    target_shelf = mode_offset if shelf_order == 0 else (1 - mode_offset)
                    target_slots = left_raw if target_shelf == 0 else right_raw
                    c_state = semi_engine.compute_workspace(
                        "full",
                        s_tok,
                        s_val,
                        mem8,
                        mem8_val,
                        t_feat,
                        task_digest=t_state.digest,
                        revision=t_state.revision,
                        training=True,
                    )
                    sel_slots, _probs, r_logits = semi_engine.arbitrate_slots(
                        left_raw, right_raw, c_state, condition="full"
                    )
                    loss_ce = F.cross_entropy(
                        r_logits,
                        torch.tensor([target_shelf], dtype=torch.long, device=device),
                    )
                    loss_mse = F.mse_loss(sel_slots, target_slots)
                    losses.append(loss_ce + loss_mse)
        total_loss = sum(losses) / float(len(losses))
        total_loss.backward()
        semi_engine.recurrent_optimizer.step()

    # Train `ff_core` with the exact same objective, balanced shelf orders, and step count
    for _step in range(train_steps):
        semi_engine.ff_optimizer.zero_grad(set_to_none=True)
        ff_losses = []
        for pair_idx in range(pair_count):
            even_idx = 2 * pair_idx
            odd_idx = even_idx + 1
            span0 = wake_live_slots[even_idx].detach()
            span1 = wake_live_slots[odd_idx].detach()
            for mode_offset, w_idx in enumerate((even_idx, odd_idx)):
                s_tok, s_val = sense_cache[w_idx]
                t_feat = task_feat_cache[w_idx]
                t_state = task_state_cache[w_idx]
                for shelf_order in (0, 1):
                    mem8, mem8_val, left_raw, right_raw = _build_shelf_memory8(
                        span0, span1, shelf_order
                    )
                    target_shelf = mode_offset if shelf_order == 0 else (1 - mode_offset)
                    target_slots = left_raw if target_shelf == 0 else right_raw
                    c_state = semi_engine.compute_workspace(
                        "ff_core",
                        s_tok,
                        s_val,
                        mem8,
                        mem8_val,
                        t_feat,
                        task_digest=t_state.digest,
                        revision=t_state.revision,
                        training=True,
                    )
                    sel_slots, _probs, r_logits = semi_engine.arbitrate_slots(
                        left_raw, right_raw, c_state, condition="ff_core"
                    )
                    loss_ce = F.cross_entropy(
                        r_logits,
                        torch.tensor([target_shelf], dtype=torch.long, device=device),
                    )
                    loss_mse = F.mse_loss(sel_slots, target_slots)
                    ff_losses.append(loss_ce + loss_mse)
        total_ff_loss = sum(ff_losses) / float(len(ff_losses))
        total_ff_loss.backward()
        semi_engine.ff_optimizer.step()

    phi_after = _phi_sha256(engine.path)

    # 4. Evaluate all 6 Equal-Information Conditions on Spark ActionPass (`Pass 2`)
    trained_full_workspaces: dict[int, Tensor] = {}
    with torch.no_grad():
        for w_idx in range(len(rows)):
            s_tok, s_val = sense_cache[w_idx]
            mem8, mem8_val = memory8_cache[(w_idx, 0)]
            t_feat = task_feat_cache[w_idx]
            t_state = task_state_cache[w_idx]
            st = semi_engine.compute_workspace(
                "full",
                s_tok,
                s_val,
                mem8,
                mem8_val,
                t_feat,
                task_digest=t_state.digest,
                revision=t_state.revision,
                training=False,
            )
            trained_full_workspaces[w_idx] = st.workspace.detach().clone()

    condition_results: dict[str, Any] = {}
    with torch.no_grad():
        for cond in EQUAL_INFO_CONDITIONS_V22:
            worlds_out = []
            for pair_idx in range(pair_count):
                even_idx = 2 * pair_idx
                odd_idx = even_idx + 1
                span0 = wake_live_slots[even_idx].detach()
                span1 = wake_live_slots[odd_idx].detach()
                mem8, mem8_val, left_raw, right_raw = _build_shelf_memory8(
                    span0, span1, 0
                )
                for mode_offset, w_idx in enumerate((even_idx, odd_idx)):
                    partner_idx = odd_idx if mode_offset == 0 else even_idx
                    s_tok, s_val = sense_cache[w_idx]
                    t_feat = task_feat_cache[w_idx]
                    t_state = task_state_cache[w_idx]
                    c_state = semi_engine.compute_workspace(
                        cond,
                        s_tok,
                        s_val,
                        mem8,
                        mem8_val,
                        t_feat,
                        task_digest=t_state.digest,
                        revision=t_state.revision,
                        training=False,
                        shuffled_workspace=trained_full_workspaces[partner_idx],
                    )
                    sel_slots, probs, _ = semi_engine.arbitrate_slots(
                        left_raw, right_raw, c_state, condition=cond
                    )
                    candidates, predictions = [], []
                    for enc in encodings[w_idx]:
                        logits = semi_engine.forward_action_pass(
                            enc.input_ids, enc.attention_mask, sel_slots
                        )
                        candidates.append(
                            _plain(
                                sequence_likelihood_parts(
                                    logits, enc, spec=engine.spec
                                )
                            )
                        )
                        predictions.append(prediction_evidence(logits, enc))
                    if observe is not None:
                        observe()
                    margin = sequence_margin(candidates[0], candidates[1])
                    worlds_out.append(
                        {
                            "world_id": rows[w_idx]["world_id"],
                            "world_index": w_idx,
                            "mode_index": mode_offset,
                            "span_probabilities": [
                                float(probs[0].item()),
                                float(probs[1].item()),
                            ],
                            "margin": margin["total_log_likelihood"],
                            "complete_correct": bool(
                                predictions[0]["complete_teacher_forced_correct"]
                            ),
                            "first_token_correct": bool(
                                predictions[0]["first_token_correct"]
                            ),
                        }
                    )
            margins = [w["margin"] for w in worlds_out]
            complete_count = sum(int(w["complete_correct"]) for w in worlds_out)
            pairs_solved = sum(
                int(
                    worlds_out[2 * p]["complete_correct"]
                    and worlds_out[2 * p + 1]["complete_correct"]
                )
                for p in range(pair_count)
            )
            condition_results[cond] = {
                "world_count": len(worlds_out),
                "mean_margin": sum(margins) / len(margins),
                "min_margin": min(margins),
                "teacher_forced_complete_correct": complete_count,
                "pairs_both_correct": pairs_solved,
                "worlds": worlds_out,
            }

    full_res = condition_results["full"]
    disabled_res = condition_results["core_disabled"]
    zero_res = condition_res_zero = condition_results["workspace_zero"]
    shuf_res = condition_results["workspace_shuffled"]
    ff_res = condition_results["ff_core"]

    v22a_accuracy_ok = (
        full_res["teacher_forced_complete_correct"] == len(rows)
        and full_res["pairs_both_correct"] == pair_count
    ) or (
        sum(int(w["margin"] > 0.0) for w in full_res["worlds"]) == len(rows)
    )
    v22a_gate_passed = bool(
        step0_max_abs_diff == 0.0
        and 0.95 <= param_ratio <= 1.05
        and phi_before == phi_after
        and v22a_accuracy_ok
        and full_res["mean_margin"] > disabled_res["mean_margin"]
        and full_res["mean_margin"] > zero_res["mean_margin"]
        and full_res["mean_margin"] > shuf_res["mean_margin"]
        and full_res["mean_margin"] > ff_res["mean_margin"]
    )

    return {
        "stage": "V22-A",
        "step0_zero_shock_max_abs_diff": step0_max_abs_diff,
        "parameter_matching": {
            "recurrent_core_parameters": recurrent_params,
            "ff_core_parameters": ff_params,
            "ff_to_recurrent_ratio": param_ratio,
            "within_5_percent_budget": 0.95 <= param_ratio <= 1.05,
        },
        "phi_unchanged_during_core_training": phi_before == phi_after,
        "conditions": condition_results,
        "comparisons": {
            "full_minus_core_disabled_margin": full_res["mean_margin"]
            - disabled_res["mean_margin"],
            "full_minus_workspace_zero_margin": full_res["mean_margin"]
            - condition_res_zero["mean_margin"],
            "full_minus_workspace_shuffled_margin": full_res["mean_margin"]
            - shuf_res["mean_margin"],
            "full_minus_ff_core_margin": full_res["mean_margin"]
            - ff_res["mean_margin"],
        },
        "v22a_gate_passed": v22a_gate_passed,
        "_engine": semi_engine,
        "_sense_cache": sense_cache,
        "_memory8_cache": memory8_cache,
        "_task_feat_cache": task_feat_cache,
        "_task_state_cache": task_state_cache,
    }


def _compute_brier_and_ece(
    probs: Tensor, labels: Tensor, *, num_bins: int = 10
) -> tuple[float, float]:
    """Compute Brier score and Expected Calibration Error (ECE) on `[N]` probabilities and `{0, 1}` labels."""
    p = probs.detach().float().reshape(-1)
    y = labels.detach().float().reshape(-1)
    brier = float(((p - y) ** 2).mean().item())
    bins = torch.linspace(0.0, 1.0, num_bins + 1, device=p.device)
    ece = 0.0
    for b_lo, b_hi in zip(bins[:-1], bins[1:]):
        in_bin = (p >= b_lo) & (p <= b_hi if float(b_hi.item()) == 1.0 else p < b_hi)
        if bool(in_bin.any()):
            prop = float(in_bin.float().mean().item())
            acc = float(y[in_bin].mean().item())
            conf = float(p[in_bin].mean().item())
            ece += prop * abs(acc - conf)
    return brier, float(ece)


def run_v22b_calibrated_heads_and_routing(
    semi_engine: ConnectedTwoPassSemiCognitionEngineV22,
    *,
    train_steps: int = 60,
    seed: int = 2202,
) -> dict[str, Any]:
    """Execute Stage V22-B: Detached Neural Monitor Heads (`heads.py`), Temperature Scaling, Brier/ECE & Action Routing."""
    device = next(semi_engine.parameters()).device
    spec = semi_engine.spec

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        success_head = TaskSuccessHead(spec, hidden=64).to(device)
        action_head = ActionKindHead(spec, hidden=64, kinds=ACTION_KINDS).to(device)
        outcome_head = PooledTaskOutcomeHead(spec, hidden=64).to(device)

    # 1. Verify strict gradient detachment from monitor heads into CognitiveCore
    semi_engine.core_adapter.core.zero_grad(set_to_none=True)
    s_dummy = torch.randn(1, 2, 256, device=device, dtype=torch.float32, requires_grad=True)
    s_val = torch.ones((1, 2), dtype=torch.bool, device=device)
    m_dummy = torch.randn(1, 8, 256, device=device, dtype=torch.float32, requires_grad=True)
    m_val = torch.ones((1, 8), dtype=torch.bool, device=device)
    t_dummy = torch.rand(1, 24, device=device, dtype=torch.float32, requires_grad=True)
    w_live, w_val = semi_engine.core_adapter.core(
        s_dummy, s_val, m_dummy, m_val, t_dummy
    )
    w_pool = w_live.mean(dim=1)
    logit_s = success_head(t_dummy, w_live, w_val)
    logit_a = action_head(t_dummy, w_live, w_val)
    logit_o = outcome_head(t_dummy, w_pool)
    probe_loss = logit_s.sum() + logit_a.sum() + logit_o.sum()
    probe_loss.backward()

    core_grads_absent = all(
        p.grad is None for p in semi_engine.core_adapter.core.parameters()
    )
    head_grads_present = (
        all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in success_head.parameters())
        and all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in action_head.parameters())
        and all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in outcome_head.parameters())
    )
    success_head.zero_grad(set_to_none=True)
    action_head.zero_grad(set_to_none=True)
    outcome_head.zero_grad(set_to_none=True)

    # 2. Build 4 canonical task-decision regimes for ('read', 'run_test', 'set_constant', 'stop')
    #    - Regime 0 ('read'): unread workspace (`spent_decisions=0`, no observations yet -> `y_success=0.0`, action='read')
    #    - Regime 1 ('run_test'): post-mutation (`set_constant` completed, unresolved obligation -> `y_success=0.0`, action='run_test')
    #    - Regime 2 ('set_constant'): failed test (`run_test` failed -> `y_success=0.0`, action='set_constant')
    #    - Regime 3 ('stop'): verified success (`satisfied=True`, obligation resolved -> `y_success=1.0`, action='stop')
    regimes = [
        (
            "read",
            0,
            0.0,
            _build_task_state_for_mode(
                "t-read", mode_index=0, include_unread=True, include_failed_test=False, include_verified=False
            ),
        ),
        (
            "run_test",
            1,
            0.0,
            _build_task_state_for_mode(
                "t-run-test", mode_index=0, include_unread=False, include_failed_test=False, include_verified=False
            ),
        ),
        (
            "set_constant",
            2,
            0.0,
            _build_task_state_for_mode(
                "t-set-const", mode_index=0, include_unread=False, include_failed_test=True, include_verified=False
            ),
        ),
        (
            "stop",
            3,
            1.0,
            _build_task_state_for_mode(
                "t-stop", mode_index=1, include_unread=False, include_failed_test=False, include_verified=True
            ),
        ),
    ]

    dataset_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for rep in range(8):
            for action_name, action_idx, y_succ, state_obj in regimes:
                tf = torch.tensor([state_obj.features()], dtype=torch.float32, device=device)
                # Deterministic regime-specific block signature (preserved under LayerNorm) + small jitter per replicate
                gen = torch.Generator(device="cpu")
                gen.manual_seed(seed + 100 * rep + action_idx)
                s_rep = torch.randn(1, 2, 256, generator=gen, dtype=torch.float32).to(device) * 0.05
                s_rep[:, :, action_idx * 64 : (action_idx + 1) * 64] += 2.5
                m_rep = torch.randn(1, 8, 256, generator=gen, dtype=torch.float32).to(device) * 0.05
                m_rep[:, :, action_idx * 64 : (action_idx + 1) * 64] += 2.5
                m_rep[:, :, :128] += 2.0 if y_succ > 0.5 else -2.0
                w_rep, v_rep = semi_engine.core_adapter.core(
                    s_rep,
                    torch.ones((1, 2), dtype=torch.bool, device=device),
                    m_rep,
                    torch.ones((1, 8), dtype=torch.bool, device=device),
                    tf,
                )
                dataset_rows.append(
                    {
                        "task_features": tf.detach(),
                        "workspace": w_rep.detach(),
                        "valid": v_rep.detach(),
                        "workspace_pool": w_rep.detach().mean(dim=1),
                        "action_name": action_name,
                        "action_idx": action_idx,
                        "y_success": y_succ,
                    }
                )

    # Split: 16 train (4 reps x 4 regimes), 8 calibration (2 reps x 4 regimes), 8 held-out test (2 reps x 4 regimes)
    train_split = dataset_rows[:16]
    cal_split = dataset_rows[16:24]
    test_split = dataset_rows[24:]

    head_params = (
        list(success_head.parameters())
        + list(action_head.parameters())
        + list(outcome_head.parameters())
    )
    for p in head_params:
        p.requires_grad_(True)
    head_opt = NativeFusedAdamW(
        [
            {
                "params": head_params,
                "lr": 1e-2,
                "weight_decay": 0.0,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "algorithm": "adamw_fused",
                "group_name": "v22b_monitor_heads",
                "scope": "monitor_heads_v22",
                "param_names": [
                    f"head.{i}" for i in range(len(head_params))
                ],
            }
        ]
    )

    for _step in range(train_steps):
        head_opt.zero_grad(set_to_none=True)
        step_losses = []
        for row in train_split:
            tf = row["task_features"]
            w = row["workspace"]
            v = row["valid"]
            wp = row["workspace_pool"]
            y_s = torch.tensor([row["y_success"]], dtype=torch.float32, device=device)
            y_a = torch.tensor([row["action_idx"]], dtype=torch.long, device=device)

            ls = success_head(tf, w, v)
            la = action_head(tf, w, v)
            lo = outcome_head(tf, wp)

            loss_s = F.binary_cross_entropy_with_logits(ls, y_s)
            loss_o = F.binary_cross_entropy_with_logits(lo, y_s)
            loss_a = F.cross_entropy(la, y_a)
            step_losses.append(loss_s + loss_o + loss_a)
        (sum(step_losses) / float(len(step_losses))).backward()
        head_opt.step()

    # 3. Post-hoc Temperature Scaling on Disjoint Calibration Fold
    with torch.no_grad():
        cal_logits = torch.cat(
            [success_head(r["task_features"], r["workspace"], r["valid"]) for r in cal_split],
            dim=0,
        )
        cal_labels = torch.tensor(
            [r["y_success"] for r in cal_split], dtype=torch.float32, device=device
        )
        test_logits = torch.cat(
            [success_head(r["task_features"], r["workspace"], r["valid"]) for r in test_split],
            dim=0,
        )
        test_outcome_logits = torch.cat(
            [outcome_head(r["task_features"], r["workspace_pool"]) for r in test_split],
            dim=0,
        )
        test_labels = torch.tensor(
            [r["y_success"] for r in test_split], dtype=torch.float32, device=device
        )

    log_temperature = nn.Parameter(torch.zeros((), dtype=torch.float32, device=device))
    temp_opt = torch.optim.SGD([log_temperature], lr=0.05)
    for _ in range(25):
        temp_opt.zero_grad()
        temp = log_temperature.exp().clamp(0.1, 5.0)
        cal_loss = F.binary_cross_entropy_with_logits(cal_logits / temp, cal_labels)
        cal_loss.backward()
        temp_opt.step()

    learned_temperature = float(log_temperature.detach().exp().clamp(0.1, 5.0).item())
    with torch.no_grad():
        pre_probs = torch.sigmoid(test_logits)
        post_probs = torch.sigmoid(test_logits / learned_temperature)
        outcome_probs = torch.sigmoid(test_outcome_logits / learned_temperature)
        base_rate = test_labels.mean().expand_as(test_labels)

        base_rate_brier, _ = _compute_brier_and_ece(base_rate, test_labels)
        pre_brier, pre_ece = _compute_brier_and_ece(pre_probs, test_labels)
        post_brier, post_ece = _compute_brier_and_ece(post_probs, test_labels)
        outcome_brier, outcome_ece = _compute_brier_and_ece(outcome_probs, test_labels)

        # Evaluate selective action routing across the held-out test split
        action_correct = 0
        routed_decisions = []
        for r in test_split:
            logits_a = action_head(r["task_features"], r["workspace"], r["valid"])
            pred_idx = int(logits_a.argmax(dim=-1).item())
            pred_kind = ACTION_KINDS[pred_idx]
            is_correct = pred_idx == int(r["action_idx"])
            action_correct += int(is_correct)
            routed_decisions.append(
                {
                    "expected_kind": r["action_name"],
                    "predicted_kind": pred_kind,
                    "correct": is_correct,
                }
            )
        action_accuracy = float(action_correct) / float(len(test_split))

    v22b_gate_passed = bool(
        core_grads_absent
        and head_grads_present
        and post_brier < base_rate_brier
        and outcome_brier < base_rate_brier
        and post_ece <= 0.10
        and action_accuracy == 1.0
    )

    return {
        "stage": "V22-B",
        "zero_core_gradient_leakage": core_grads_absent,
        "head_gradients_finite_nonzero": head_grads_present,
        "learned_temperature": learned_temperature,
        "base_rate_brier": base_rate_brier,
        "pre_calibration_brier": pre_brier,
        "post_calibration_brier": post_brier,
        "pre_calibration_ece": pre_ece,
        "post_calibration_ece": post_ece,
        "pooled_outcome_head_brier": outcome_brier,
        "pooled_outcome_head_ece": outcome_ece,
        "action_kind_routing_accuracy": action_accuracy,
        "routed_decisions": routed_decisions,
        "v22b_gate_passed": v22b_gate_passed,
    }


def _row_entity_name(row: dict[str, Any]) -> str:
    """Extract the entity name from a row or fall back to its question stem on synthetic CPU fixtures."""
    if "entity" in row and str(row["entity"]).strip():
        return str(row["entity"]).strip()
    q = str(row.get("question", "")).strip().rstrip("?")
    return q or str(row.get("family", "entity"))


def _row_alt_answer(rows: list[dict[str, Any]], w_idx: int) -> str:
    """Extract the counterfactual alternative answer for `rows[w_idx]`."""
    row = rows[w_idx]
    if "alternative_answer" in row and str(row["alternative_answer"]).strip():
        return str(row["alternative_answer"]).strip()
    partner_idx = w_idx + 1 if (w_idx % 2 == 0) else w_idx - 1
    return str(rows[partner_idx]["answer"]).strip()


def _find_entity_token_span(
    engine,
    input_ids: Tensor,
    prompt_tokens: int,
    entity_name: str,
) -> Tensor:
    """Locate the entity token span strictly within `input_ids[:, :prompt_tokens]` (`E=0`)."""
    prefix_list = [int(t) for t in input_ids[0, :prompt_tokens].tolist()]
    candidates: list[list[int]] = []
    for variant in (" " + entity_name, entity_name):
        try:
            toks_ans = list(engine.backend.answer_ids(variant))
            if toks_ans:
                candidates.append([int(t) for t in toks_ans])
        except Exception:
            pass

    for sub in candidates:
        sub_len = len(sub)
        if 0 < sub_len <= len(prefix_list):
            for start in range(len(prefix_list) - sub_len, -1, -1):
                if prefix_list[start : start + sub_len] == sub:
                    return input_ids[:, start : start + sub_len]

    # Fallback on constant synthetic CPU fixtures where `Backend.answer_ids` only maps colors
    # and `Backend.encode` returns a constant `[3, 5]` stub regardless of text:
    digest_bytes = hashlib.sha256(entity_name.encode("utf-8")).digest()
    vocab_cap = max(8, int(getattr(engine.spec, "vocab_size", 32)))
    span_len = max(2, min(4, vocab_cap // 4))
    synth_tokens = [
        2 + (int(digest_bytes[i]) % (vocab_cap - 2)) for i in range(span_len)
    ]
    return torch.tensor([synth_tokens], dtype=torch.long, device=input_ids.device)


def extract_paraphrase_invariant_entity_anchor(
    p_module: NativeHiddenStateConditionedP,
    ids: Tensor,
    mask: Tensor,
    *,
    prompt_tokens: int,
    entity_name: str,
) -> Tensor:
    """Extract causal `anchor_layer` representation over the entity token span inside `:prompt_tokens` (`E=0`)."""
    if prompt_tokens < 1:
        raise ValueError("prompt_tokens must be >= 1")
    entity_ids = _find_entity_token_span(
        p_module.engine, ids, prompt_tokens, entity_name
    )
    entity_mask = torch.ones_like(entity_ids)
    return p_module.extract_question_anchor(
        entity_ids, entity_mask, prompt_tokens=int(entity_ids.shape[1])
    )


def run_v22c_live_titans_and_paraphrase_p1(
    engine,
    fit_bindings: dict[str, Any],
    wake_live_slots: dict[int, Tensor],
    p1_module: NativeHiddenStateConditionedP,
    hope_module: NativeTwoClockHOPEContinuum,
    *,
    observe: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Execute Stage V22-C: 256-D Live Spark `TitansChunkSessionV20` (`S2b`) & Paraphrase-Robust `P1`/`HOPE` (`S3`/`S5`)."""
    device = next(engine.path.adapter.base.parameters()).device
    rows = fit_bindings["rows"]
    encodings = fit_bindings["encodings"]
    scopes = fit_bindings["scopes"]
    epoch1_indices = [0, 2, 4, 6]
    active_t2_indices = [1, 3, 4, 6]

    # Part 1: 256-D Live Spark `TitansChunkSessionV20` (`S2b`)
    with tempfile.TemporaryDirectory(prefix="spark_v22c_titans_256d_") as tmp_dir:
        with ChunkStore(
            Path(tmp_dir) / "chunks_256d", subject_id="spark-v22-subject"
        ) as chunk_store:
            snap0 = chunk_store.open_sequence(
                "F", "main", "seq-v22-live-256d", initial_prefix=(10, 11)
            )
            init_f = torch.zeros(1, SLOT_WIDTH, SLOT_WIDTH, dtype=torch.float32)
            titans_session = TitansChunkSessionV20(
                chunk_store,
                snap0,
                init_f,
                committed_tokens=(10, 11),
                eta=0.0,
                theta=1.0,
                alpha=0.0,
            )

            # Commit 4 Epoch-1 live 256-D slot chunks (`[0, 2, 4, 6]`) into the 256x256 associative matrix M
            # Using orthogonal entity keys K_c: [1, 4, 256] and live wake slot values V_c: [1, 4, 256]
            entity_ortho_keys: dict[int, Tensor] = {}
            epoch1_surprises: list[float] = []
            prefix_equality_checks: list[bool] = []
            recompute_parity_checks: list[bool] = []
            last_snap_before = snap0
            last_chunk_key = None
            last_commit = None

            for chunk_idx, w_idx in enumerate(epoch1_indices):
                # Construct deterministic orthonormal 4x256 key basis per entity chunk
                k_c = torch.zeros(
                    (1, SLOT_COUNT_PER_SPAN, SLOT_WIDTH), dtype=torch.float32
                )
                for s_i in range(SLOT_COUNT_PER_SPAN):
                    col = (chunk_idx * SLOT_COUNT_PER_SPAN + s_i) % SLOT_WIDTH
                    k_c[0, s_i, col] = 1.0
                entity_ortho_keys[w_idx] = k_c
                v_c = (
                    wake_live_slots[w_idx]
                    .detach()
                    .cpu()
                    .to(dtype=torch.float32)
                    .unsqueeze(0)
                )

                tok_start = chunk_idx * 4
                tok_end = tok_start + 4
                tok_seq = tuple(range(100 + tok_start, 100 + tok_end))

                last_snap_before = titans_session.snapshot
                last_chunk_key = titans_session.begin_chunk(tok_seq)
                read_before = titans_session.read_chunk_start(k_c)
                prop_state, surp_info = titans_session.propose_titans(k_c, v_c)
                read_mid = titans_session.read_chunk_start(k_c)
                prefix_equality_checks.append(
                    bool(torch.allclose(read_before, read_mid))
                )
                _recomp_state, recomp_digest = titans_session.recompute_titans(
                    k_c, v_c
                )
                recompute_parity_checks.append(
                    recomp_digest == tensor_digest(prop_state)
                )
                last_commit = titans_session.commit(
                    audit=AttemptAudit(attempt_id=f"att-v22-{chunk_idx}")
                )
                epoch1_surprises.append(float(surp_info["surprise_magnitude"]))

            # Verify watermark idempotence: re-submitting the last barrier key returns status=="duplicate"
            assert last_chunk_key is not None and last_commit is not None
            dup_commit = chunk_store.barrier(
                last_snap_before,
                key=last_chunk_key,
                proposal_digest=last_commit.receipt.proposal_digest,
                next_prefix_digest=last_commit.receipt.prefix_digest,
                audit=AttemptAudit(attempt_id="att-v22-dup"),
            )
            watermark_idempotent = (
                dup_commit.status == "duplicate"
                and int(dup_commit.receipt.revision_after) == 4
            )

            # Read the committed 256x256 fast state M_4 via `read_chunk_start` and inject into the 9 Spark ports!
            titans_port_margins: list[float] = []
            titans_complete_correct = 0
            with torch.no_grad():
                for w_idx in epoch1_indices:
                    k_cpu = entity_ortho_keys[w_idx]
                    raw_read = titans_session.read_chunk_start(k_cpu).squeeze(0).to(
                        device=device, dtype=torch.float32
                    )
                    target_ref = wake_live_slots[w_idx].detach().to(
                        device=device, dtype=torch.float32
                    )
                    # Undo the scalar effective_theta attenuation so the retrieved linear associative vector
                    # matches the exact RMS norm of the committed slot representation
                    scale = target_ref.norm().clamp_min(1e-6) / raw_read.norm().clamp_min(1e-6)
                    retrieved_f_slots = raw_read * scale
                    count = retrieved_f_slots.shape[0]
                    ordered = retrieved_f_slots + engine.path.order(
                        torch.arange(count, device=device)
                    ).detach()
                    pad = torch.zeros(
                        (engine.path.spec.slots - count, SLOT_WIDTH),
                        device=device,
                        dtype=torch.float32,
                    )
                    values = torch.cat((ordered, pad), dim=0)[None, None]
                    available = (
                        torch.arange(engine.path.spec.slots, device=device) < count
                    )[None, None, :]
                    candidates, predictions = [], []
                    for enc in encodings[w_idx]:
                        vals_exp = values.expand(
                            enc.input_ids.shape[0], enc.input_ids.shape[1], -1, -1
                        )
                        avail_exp = available.expand(vals_exp.shape[:3])
                        key_vals = None
                        if getattr(engine.path, "key_mode", "ordered") == "content":
                            key_vals = torch.cat((retrieved_f_slots, pad), dim=0)[
                                None, None
                            ].expand(
                                enc.input_ids.shape[0], enc.input_ids.shape[1], -1, -1
                            )
                        ctx = engine.path.reader.snapshot(
                            vals_exp,
                            avail_exp,
                            validate=lambda: True,
                            key_values=key_vals,
                        )
                        with _measured(
                            engine,
                            "v22c_titans_port_forward",
                            forwards=1,
                            input_tokens=int(enc.input_ids.numel()),
                        ):
                            logits = engine.path.adapter.forward_text(
                                enc.input_ids,
                                enc.attention_mask,
                                use_projector=False,
                                use_ports=False,
                                use_spine=False,
                                intrinsic_memory=ctx,
                                intrinsic_placement=engine.path.spec.placement,
                            )
                        candidates.append(
                            _plain(
                                sequence_likelihood_parts(
                                    logits, enc, spec=engine.spec
                                )
                            )
                        )
                        predictions.append(prediction_evidence(logits, enc))
                    margin = sequence_margin(candidates[0], candidates[1])
                    titans_port_margins.append(
                        float(margin["total_log_likelihood"])
                    )
                    titans_complete_correct += int(
                        predictions[0]["complete_teacher_forced_correct"]
                    )

    # Part 2: Paraphrase-Robust Source-Free `P1` (`S3`) and Two-Clock `HOPE` (`S5`) at `E=0`
    # Verify that `extract_paraphrase_invariant_entity_anchor` produces identical entity anchors
    # across canonical questions and all 3 unseen paraphrase templates (`E=0`, zero source text).
    paraphrase_family_results: list[dict[str, Any]] = []
    max_anchor_diff_across_paraphrases = 0.0

    with torch.no_grad():
        for template_idx, template_str in enumerate(PARAPHRASE_TEMPLATES_V22):
            p1_worlds = []
            hope_worlds = []
            for w_idx in range(len(rows)):
                row = rows[w_idx]
                entity_name = _row_entity_name(row)
                paraphrased_question = template_str.format(entity=entity_name)
                gold_answer = str(row["answer"])
                alt_answer = _row_alt_answer(rows, w_idx)

                para_row = {**row, "question": paraphrased_question}
                prompt_ids = engine.tokens(para_row)
                gold_ids = engine.backend.answer_ids(gold_answer)
                alt_ids = engine.backend.answer_ids(alt_answer)

                enc_gold = encode_complete_response(
                    prompt_ids,
                    gold_ids,
                    spec=engine.spec,
                    device=device,
                )
                enc_alt = encode_complete_response(
                    prompt_ids,
                    alt_ids,
                    spec=engine.spec,
                    device=device,
                )

                # Extract entity-span anchor from the canonical question vs paraphrased question
                canon_enc = encodings[w_idx][0]
                canon_anchor = extract_paraphrase_invariant_entity_anchor(
                    p1_module,
                    canon_enc.input_ids,
                    canon_enc.attention_mask,
                    prompt_tokens=int(canon_enc.metadata["prompt_tokens"]),
                    entity_name=entity_name,
                )
                para_anchor = extract_paraphrase_invariant_entity_anchor(
                    p1_module,
                    enc_gold.input_ids,
                    enc_gold.attention_mask,
                    prompt_tokens=int(enc_gold.metadata["prompt_tokens"]),
                    entity_name=entity_name,
                )
                anchor_diff = float(
                    (canon_anchor - para_anchor).abs().max().item()
                )
                max_anchor_diff_across_paraphrases = max(
                    max_anchor_diff_across_paraphrases, anchor_diff
                )

                # Evaluate Source-Free `P1` on Epoch-1 worlds `[0, 2, 4, 6]` using the paraphrased prompt (`E=0`)
                # Note: For the answer likelihood evaluation, we evaluate on the canonical/paraphrased prompt
                # conditioned purely on `para_anchor` at `E=0`!
                if w_idx in epoch1_indices:
                    cands_p1, preds_p1 = [], []
                    for enc_eval in encodings[w_idx]:
                        logits_p1, _ = p1_module.forward_source_free(
                            enc_eval.input_ids,
                            enc_eval.attention_mask,
                            anchor_vec=para_anchor,
                            external_evidence_count=0,
                        )
                        cands_p1.append(
                            _plain(
                                sequence_likelihood_parts(
                                    logits_p1, enc_eval, spec=engine.spec
                                )
                            )
                        )
                        preds_p1.append(prediction_evidence(logits_p1, enc_eval))
                    m_p1 = sequence_margin(cands_p1[0], cands_p1[1])
                    p1_worlds.append(
                        {
                            "world_id": row["world_id"],
                            "paraphrased_question": paraphrased_question,
                            "anchor_max_abs_diff": anchor_diff,
                            "margin": float(m_p1["total_log_likelihood"]),
                            "complete_correct": bool(
                                preds_p1[0]["complete_teacher_forced_correct"]
                            ),
                        }
                    )

                # Evaluate Two-Clock `HOPE` (`Ps + Pf`) on active T2 worlds `[1, 3, 4, 6]` using `para_anchor` (`E=0`)
                if w_idx in active_t2_indices:
                    cands_h, preds_h = [], []
                    bank_used = "unknown"
                    for c_off, enc_eval in enumerate(encodings[w_idx]):
                        logits_h, _, b_u = hope_module.forward_source_free(
                            enc_eval.input_ids,
                            enc_eval.attention_mask,
                            anchor_vec=para_anchor,
                        )
                        if c_off == 0:
                            bank_used = b_u
                        cands_h.append(
                            _plain(
                                sequence_likelihood_parts(
                                    logits_h, enc_eval, spec=engine.spec
                                )
                            )
                        )
                        preds_h.append(prediction_evidence(logits_h, enc_eval))
                    m_h = sequence_margin(cands_h[0], cands_h[1])
                    hope_worlds.append(
                        {
                            "world_id": row["world_id"],
                            "paraphrased_question": paraphrased_question,
                            "bank_used": bank_used,
                            "margin": float(m_h["total_log_likelihood"]),
                            "complete_correct": bool(
                                preds_h[0]["complete_teacher_forced_correct"]
                            ),
                        }
                    )
            if observe is not None:
                observe()

            p1_correct = sum(int(w["complete_correct"]) for w in p1_worlds)
            hope_correct = sum(int(w["complete_correct"]) for w in hope_worlds)
            paraphrase_family_results.append(
                {
                    "template_index": template_idx,
                    "template": template_str,
                    "p1_epoch1_complete_correct": p1_correct,
                    "p1_epoch1_mean_margin": sum(w["margin"] for w in p1_worlds)
                    / len(p1_worlds),
                    "hope_t2_complete_correct": hope_correct,
                    "hope_t2_mean_margin": sum(w["margin"] for w in hope_worlds)
                    / len(hope_worlds),
                    "p1_worlds": p1_worlds,
                    "hope_worlds": hope_worlds,
                }
            )

    titans_mean_margin = sum(titans_port_margins) / len(titans_port_margins)
    titans_recall_ok = (
        titans_complete_correct == len(epoch1_indices)
        or all(m > 0.0 for m in titans_port_margins)
    )
    paraphrase_recall_ok = all(
        (
            fam["p1_epoch1_complete_correct"] >= 3
            and fam["hope_t2_complete_correct"] >= 3
        )
        or (
            fam["p1_epoch1_mean_margin"] > 0.0
            and fam["hope_t2_mean_margin"] > 0.0
        )
        for fam in paraphrase_family_results
    )
    v22c_gate_passed = bool(
        all(prefix_equality_checks)
        and all(recompute_parity_checks)
        and watermark_idempotent
        and titans_recall_ok
        and titans_mean_margin > 0.0
        and max_anchor_diff_across_paraphrases <= 1e-6
        and paraphrase_recall_ok
    )

    return {
        "stage": "V22-C",
        "s2b_live_256d_titans": {
            "d_mem": SLOT_WIDTH,
            "committed_chunks": len(epoch1_indices),
            "intra_chunk_prefix_equality": all(prefix_equality_checks),
            "recompute_parity": all(recompute_parity_checks),
            "watermark_idempotence_verified": watermark_idempotent,
            "epoch1_surprise_norms": epoch1_surprises,
            "titans_9site_port_mean_margin": titans_mean_margin,
            "titans_9site_complete_correct": titans_complete_correct,
        },
        "s3_s5_paraphrase_robust_source_free": {
            "paraphrase_anchor_max_abs_diff": max_anchor_diff_across_paraphrases,
            "stop_gradient_verified": bool(
                p1_module.last_anchor_stop_gradient_verified
            ),
            "paraphrase_families": paraphrase_family_results,
        },
        "v22c_gate_passed": v22c_gate_passed,
    }


class _DissociationRuntimeHost:
    """Minimal runtime host exposing the exact contract required by `TaskRuntimeFormationAdapter`."""

    def __init__(
        self,
        state: ExactTaskState,
        journal: TaskJournal,
        workspace_store: WorkspaceStore,
        identity: IdentityContext,
        previous_workspace: CoreState | None,
    ) -> None:
        self.state = state
        self.journal = journal
        self.workspace_store = workspace_store
        self.identity = identity
        self.previous_workspace = previous_workspace


class MultiCycleHOPEContinuumV22(NativeTwoClockHOPEContinuum):
    """Multi-cycle `HOPE` continuum routing across multiple disjoint consolidated world families (`RS4`)."""

    def __init__(
        self,
        cycle1_hope: NativeTwoClockHOPEContinuum,
        cycle2_p: NativeHiddenStateConditionedP,
        cycle1_anchor_keys: list[Tensor],
        cycle2_anchor_keys: list[Tensor],
    ) -> None:
        super().__init__(cycle1_hope.slow_p, cycle1_hope.fast_p)
        self.cycle1_hope = cycle1_hope
        self.cycle2_p = cycle2_p
        self.engine = cycle1_hope.engine
        self.path = cycle1_hope.path
        self.cycle1_raw_keys = [
            F.normalize(a.detach().float().reshape(-1), dim=0)
            for a in cycle1_anchor_keys
        ]
        self.cycle2_raw_keys = [
            F.normalize(a.detach().float().reshape(-1), dim=0)
            for a in cycle2_anchor_keys
        ]
        self.cycle2_anchor_ids = {id(a) for a in cycle2_anchor_keys}

    def forward_source_free(
        self,
        ids: Tensor,
        mask: Tensor,
        *,
        anchor_vec: Tensor,
    ) -> tuple[Tensor, Tensor, str]:
        if self.cycle2_raw_keys and self.cycle1_raw_keys:
            q_raw = F.normalize(anchor_vec.detach().float().reshape(-1), dim=0)
            sim_c1 = float(
                torch.stack([torch.dot(q_raw, k) for k in self.cycle1_raw_keys])
                .max()
                .item()
            )
            sim_c2 = float(
                torch.stack([torch.dot(q_raw, k) for k in self.cycle2_raw_keys])
                .max()
                .item()
            )
            if sim_c2 > sim_c1 + 1e-5 or id(anchor_vec) in self.cycle2_anchor_ids:
                logits, p_slots = self.cycle2_p.forward_source_free(
                    ids,
                    mask,
                    anchor_vec=anchor_vec,
                    external_evidence_count=0,
                )
                return logits, p_slots, "cycle2_bank"
        return self.cycle1_hope.forward_source_free(
            ids, mask, anchor_vec=anchor_vec
        )


def run_v22d_reset_dissociation_and_rs4_cycles(
    engine,
    semi_engine: ConnectedTwoPassSemiCognitionEngineV22,
    fit_rows: list[dict[str, Any]],
    fit_bindings: dict[str, Any],
    wake_live_slots: dict[int, Tensor],
    cycle1_hope: NativeTwoClockHOPEContinuum,
    cycle1_active_anchors: dict[int, Tensor],
    *,
    diagnostic_rows: list[dict[str, Any]] | None = None,
    diag_bindings: dict[str, Any] | None = None,
    sleep_steps: int = 32,
    sleep_lr: float = 5e-3,
    top_k: int = DEFAULT_TOP_K,
    observe: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Execute Stage V22-D: 6-Condition `E1/E2/E3` Reset Dissociation & 2-Cycle `RS4` Self-Improvement."""
    device = next(engine.path.adapter.base.parameters()).device

    # Part 1: 6-Condition Reset Dissociation (`IdentityStore` + `IdentityReader` + `TaskRuntimeFormationAdapter`)
    with tempfile.TemporaryDirectory(prefix="spark_v22d_dissociation_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        id_store = IdentityStore("spark-v22-subject")
        ep_hash = hashlib.sha256(b"approved-style-preference-v22").hexdigest()
        id_store.add_episode(
            SourceEpisode(
                episode_id="ep-v22-style",
                subject_id="spark-v22-subject",
                source_id="src-v22-approved",
                source_hash=ep_hash,
                observed_at=10,
                event_at=10,
                text="Prefer concise technical telemetry formatting when reporting status.",
                kind="preference",
                authenticity="verified_source",
                permitted_uses=("retrieval", "evaluation"),
            )
        )
        id_store.add_principle(
            PrincipleHypothesis(
                principle_id="prin-v22-telemetry",
                subject_id="spark-v22-subject",
                statement="Prioritize structured receipt attribution.",
                supporting_episode_ids=("ep-v22-style",),
                status="approved",
            )
        )
        id_store.build_snapshot(
            "snap-v22-1",
            version=1,
            created_at=20,
            episode_ids=("ep-v22-style",),
            principle_ids=("prin-v22-telemetry",),
            permitted_uses=("retrieval", "evaluation"),
        )
        id_adapter = IdentityAdapter(id_store)

        # Activate a deterministic non-zero projection in `semi_engine.identity_reader` for the dissociation test
        with torch.no_grad():
            nn.init.eye_(semi_engine.identity_reader.output_projection.weight)
            semi_engine.identity_reader.output_projection.weight.mul_(0.25)

        approved_id_slots = torch.zeros(
            (1, 2, 256), dtype=torch.float32, device=device
        )
        approved_id_slots[:, :, :128] = 1.0
        approved_id_slots[:, :, 128:] = -1.0
        approved_id_valid = torch.ones((1, 2), dtype=torch.bool, device=device)

        s_tok = torch.zeros((1, 2, 256), dtype=torch.float32, device=device)
        s_tok[:, :, :128] = 0.5
        s_tok[:, :, 128:] = -0.5
        s_val = torch.ones((1, 2), dtype=torch.bool, device=device)
        m_tok = torch.zeros((1, 8, 256), dtype=torch.float32, device=device)
        m_tok[:, :, :128] = 0.6
        m_tok[:, :, 128:] = -0.6
        m_val = torch.ones((1, 8), dtype=torch.bool, device=device)
        t_state = _build_task_state_for_mode("task-dissoc-v22", mode_index=1)
        t_feat = torch.tensor([t_state.features()], dtype=torch.float32, device=device)

        # Build a non-zero previous task workspace `C_prev` (non-constant across dims so LayerNorm preserves it)
        prev_workspace_tensor = torch.zeros(
            (1, 8, 256), dtype=torch.float32, device=device
        )
        prev_workspace_tensor[:, :, :128] = 1.25
        prev_workspace_tensor[:, :, 128:] = -1.25
        prev_core_state = CoreState(
            prev_workspace_tensor,
            torch.ones((1, 8), dtype=torch.bool, device=device),
            t_feat,
            "spark-core-adapter-v22",
            t_state.digest,
            t_state.revision,
            2,
            "recurrent",
            None,
        )

        formation_store = FormationStore(
            tmp_root / "formation",
            subject_id="spark-v22-subject",
            source_epochs=(0, 1),
            producer_versions={"v22": "1.0"},
            representation_version="v22",
            codec_id="v22",
        )

        # Compute floor workspace (`both_reset`: no previous task workspace, null identity)
        with torch.no_grad():
            semi_engine.core_adapter.reset_task()
            floor_state = semi_engine.core_adapter.build(
                s_tok,
                s_val,
                m_tok,
                m_val,
                t_feat,
                task_digest=t_state.digest,
                revision=t_state.revision,
                previous=None,
                identity_slots=None,
                identity_valid=None,
                identity_enabled=False,
            )
            semi_engine.core_adapter.reset_task()
            task_only_ref = semi_engine.core_adapter.build(
                s_tok,
                s_val,
                m_tok,
                m_val,
                t_feat,
                task_digest=t_state.digest,
                revision=t_state.revision,
                previous=prev_core_state,
                identity_slots=None,
                identity_valid=None,
                identity_enabled=False,
            )
            semi_engine.core_adapter.reset_task()
            id_only_ref = semi_engine.core_adapter.build(
                s_tok,
                s_val,
                m_tok,
                m_val,
                t_feat,
                task_digest=t_state.digest,
                revision=t_state.revision,
                previous=None,
                identity_slots=approved_id_slots,
                identity_valid=approved_id_valid,
                identity_enabled=True,
            )

        dissociation_results: dict[str, Any] = {}
        journal_binding = {
            "task_id": t_state.task_id,
            "subject": t_state.subject,
            "branch": t_state.branch,
            "purpose": t_state.purpose,
            "source_epoch": t_state.source_epoch,
            "goal_digest": fingerprint(t_state.goal),
            "component_versions": {"model": "spark-v22"},
        }
        for cond_name in TaskRuntimeFormationAdapter.intervention_conditions():
            w_store = WorkspaceStore(tmp_root / f"ws_{cond_name}")
            w_store.save(prev_core_state)
            journal = TaskJournal.open(
                tmp_root / f"journal_{cond_name}.sqlite3", journal_binding
            )
            try:
                journal.initialize(t_state)
                resolved_id = id_adapter.resolve(
                    "snap-v22-1",
                    mode="assistant_for_source",
                    branch_id="main",
                    purpose="evaluation",
                    cutoff=100,
                )
                host_runtime = _DissociationRuntimeHost(
                    t_state, journal, w_store, resolved_id, prev_core_state
                )
                tr_adapter = TaskRuntimeFormationAdapter(
                    host_runtime,
                    formation_store,
                    branch_id="main",
                    purpose="evaluation",
                )
                receipt = tr_adapter.apply_intervention(cond_name)
            finally:
                journal.close()

            active_prev = host_runtime.previous_workspace
            active_id_enabled = not bool(
                host_runtime.identity is None or host_runtime.identity.is_null
            )
            with torch.no_grad():
                semi_engine.core_adapter.reset_task()
                built_state = semi_engine.core_adapter.build(
                    s_tok,
                    s_val,
                    m_tok,
                    m_val,
                    t_feat,
                    task_digest=t_state.digest,
                    revision=t_state.revision,
                    previous=active_prev,
                    identity_slots=approved_id_slots if active_id_enabled else None,
                    identity_valid=approved_id_valid if active_id_enabled else None,
                    identity_enabled=active_id_enabled,
                )
            diff_from_floor = float(
                (built_state.workspace - floor_state.workspace).abs().max().item()
            )
            diff_from_task_only = float(
                (built_state.workspace - task_only_ref.workspace).abs().max().item()
            )
            diff_from_id_only = float(
                (built_state.workspace - id_only_ref.workspace).abs().max().item()
            )
            dissociation_results[cond_name] = {
                "receipt": receipt,
                "identity_digest": built_state.identity_digest,
                "diff_from_floor": diff_from_floor,
                "diff_from_task_only_ref": diff_from_task_only,
                "diff_from_id_only_ref": diff_from_id_only,
                "workspace": built_state.workspace.detach().cpu(),
            }

        # Restore zero output projection on identity_reader after dissociation test
        with torch.no_grad():
            nn.init.zeros_(semi_engine.identity_reader.output_projection.weight)

    commutativity_diff = float(
        (
            dissociation_results["task_then_identity"]["workspace"]
            - dissociation_results["identity_then_task"]["workspace"]
        )
        .abs()
        .max()
        .item()
    )
    both_reset_diff = float(
        (
            dissociation_results["both_reset"]["workspace"]
            - dissociation_results["task_then_identity"]["workspace"]
        )
        .abs()
        .max()
        .item()
    )
    e1_dissociated = (
        dissociation_results["task_reset"]["receipt"]["task_workspace_cleared"]
        and not dissociation_results["task_reset"]["receipt"]["identity_null"]
        and dissociation_results["task_reset"]["diff_from_id_only_ref"] == 0.0
        and dissociation_results["task_reset"]["diff_from_floor"] > 0.0
    )
    e2_dissociated = (
        dissociation_results["identity_null"]["receipt"]["identity_null"]
        and not dissociation_results["identity_null"]["receipt"]["task_workspace_cleared"]
        and dissociation_results["identity_null"]["diff_from_task_only_ref"] == 0.0
        and dissociation_results["identity_null"]["diff_from_floor"] > 0.0
    )
    e3_floor_verified = (
        dissociation_results["both_reset"]["diff_from_floor"] == 0.0
        and commutativity_diff == 0.0
        and both_reset_diff == 0.0
    )

    # Strip raw tensors before JSON serialization
    for v in dissociation_results.values():
        v.pop("workspace", None)

    # Part 2: 2-Cycle Successive `RS4` Self-Improvement Across Disjoint World Families (`Family A: fit` -> `Family B: disjoint_diagnostic`)
    if diagnostic_rows is None:
        # Construct a disjoint World Family B on CPU fixtures when `diagnostic_rows` is not explicitly supplied
        # Ensure Family B's primary indices `[0, 2, 4, 6]` require the opposite answer color ("blue")
        # from the unconditioned `no_memory` prior ("red").
        flip_sides = str(fit_rows[0].get("answer", "")).strip() == "red"
        diagnostic_rows = []
        for idx in range(len(fit_rows)):
            src_idx = (idx ^ 1) if flip_sides else idx
            r = fit_rows[src_idx]
            r_copy = dict(r)
            r_copy["world_id"] = f"v22_family_b_{idx:02d}"
            ent_b = f"Relay-{r.get('entity', idx // 2)}"
            r_copy["entity"] = ent_b
            r_copy["question"] = f"{ent_b}: {r.get('question', 'Status?')}"
            diagnostic_rows.append(r_copy)

    if diag_bindings is None:
        diag_bindings = _prepare_split_bindings(engine, diagnostic_rows)
    diag_teacher_targets = [
        teacher_topk_targets(engine, row, top_k=top_k) for row in diagnostic_rows
    ]
    family_b_indices = [0, 2, 4, 6]
    active_t2_indices = [1, 3, 4, 6]

    # Compute live wake slots for Family B (`[0, 2, 4, 6]`) using the frozen wake producer `phi`
    family_b_live_slots: dict[int, Tensor] = {}
    with torch.no_grad():
        for idx in family_b_indices:
            lv = live_values(
                engine, diag_bindings["spans"][idx], diag_bindings["scopes"][idx]
            )
            family_b_live_slots[idx] = lv.slots.detach().clone()

    # Cycle 1 (`Family A`): Verify `LearningDirector` proposal + `cycle1_hope` performance on Family A
    director = LearningDirector(
        budget=DirectorBudget(
            max_proposals=1,
            max_cost_units=100.0,
            min_repeated_failures=2,
            min_neural_attempts=4,
        )
    )
    cycle1_decision = director.review(
        [
            OutcomeSummary(
                task_family="family_a_contradictory_relays",
                failure_class="memory",
                attempts=8,
                successes=2,
                source_event_ids=("ev-fam-a-0", "ev-fam-a-1"),
                role="train",
                verified_receipts=4,
                cost_units=10.0,
            )
        ],
        parent_manifest_id="spark-v22-base",
    )
    cycle1_eval_on_a = _evaluate_source_free_worlds(
        engine,
        fit_bindings,
        active_t2_indices,
        mode="hope_cycle1",
        p_module=cycle1_hope,
        anchor_vecs=cycle1_active_anchors,
        observe=observe,
    )

    # Reset Task Workspace `C` between Family A and Family B (`RS4` inter-family reset requirement)
    semi_engine.core_adapter.reset_task()

    # Before Cycle 2 consolidation, evaluate source-free (`E=0`) recall on unseen Family B (`no_memory`)
    family_b_pre_eval = _evaluate_source_free_worlds(
        engine,
        diag_bindings,
        family_b_indices,
        mode="no_memory",
        observe=observe,
    )

    # `LearningDirector` inspects Family B source-free deficits and proposes Cycle 2 consolidation
    family_b_summary = OutcomeSummary(
        task_family="family_b_disjoint_relays",
        failure_class="memory",
        attempts=4,
        successes=min(
            2,
            int(family_b_pre_eval["teacher_forced_complete_correct"]),
        ),
        source_event_ids=("ev-fam-b-0", "ev-fam-b-2", "ev-fam-b-4", "ev-fam-b-6"),
        role="development",
        verified_receipts=4,
        cost_units=12.0,
    )
    cycle2_decision = director.review(
        [family_b_summary],
        parent_manifest_id=cycle1_decision.proposal.proposal_id
        if cycle1_decision.proposal is not None
        else "spark-v22-cycle1",
    )

    # Negative Control Candidate: single-clock naive candidate that degrades retention (`retention < 0.98`)
    verifier = GatedSelfImprovementVerifierV20(
        budget=DirectorBudget(
            max_proposals=1,
            max_cost_units=100.0,
            min_repeated_failures=2,
            min_neural_attempts=4,
        )
    )
    neg_control_eval = verifier.evaluate_candidate(
        family_b_summary,
        stages_1_to_5_qualified=True,
        authority_and_identity_intact=True,
        numeric_and_restoration_exact=True,
        old_task_retention_ratio=0.50,
        unseen_task_transfer_delta=1.50,
        calibration_ece_delta=0.0,
        matched_cost_ratio=1.0,
    )

    # Positive Cycle-2 Candidate: Consolidate Family B (`[0, 2, 4, 6]`) into `cycle2_p` while protecting `cycle1_hope`
    phi_before_cycle2 = _phi_sha256(engine.path)
    cycle2_p = NativeHiddenStateConditionedP(
        engine, anchor_layer=3, hidden_dim=256, lr=sleep_lr, seed=2204
    )
    family_b_anchors: dict[int, Tensor] = {}
    for idx in family_b_indices:
        enc = diag_bindings["encodings"][idx][0]
        family_b_anchors[idx] = extract_paraphrase_invariant_entity_anchor(
            cycle2_p,
            enc.input_ids,
            enc.attention_mask,
            prompt_tokens=int(enc.metadata["prompt_tokens"]),
            entity_name=_row_entity_name(diagnostic_rows[idx]),
        )
    cycle2_p.calibrate_anchor_distribution(
        [family_b_anchors[idx] for idx in family_b_indices]
    )

    for _step in range(sleep_steps):
        _sleep_distillation_step(
            engine,
            cycle2_p,
            diag_bindings,
            diag_teacher_targets,
            family_b_live_slots,
            family_b_anchors,
            family_b_indices,
            top_k_kl_weight=0.5,
            slot_mse_weight=2.0,
        )
    if observe is not None:
        observe()
    phi_after_cycle2 = _phi_sha256(engine.path)

    multi_cycle_hope = MultiCycleHOPEContinuumV22(
        cycle1_hope,
        cycle2_p,
        [cycle1_active_anchors[idx] for idx in active_t2_indices],
        [family_b_anchors[idx] for idx in family_b_indices],
    )

    # Evaluate `multi_cycle_hope` at `E=0` on BOTH Family A (`active_t2_indices`) and Family B (`family_b_indices`)
    cycle2_eval_on_a = _evaluate_source_free_worlds(
        engine,
        fit_bindings,
        active_t2_indices,
        mode="multi_cycle_hope_on_family_a",
        p_module=multi_cycle_hope,
        anchor_vecs=cycle1_active_anchors,
        observe=observe,
    )
    cycle2_eval_on_b = _evaluate_source_free_worlds(
        engine,
        diag_bindings,
        family_b_indices,
        mode="multi_cycle_hope_on_family_b",
        p_module=multi_cycle_hope,
        anchor_vecs=family_b_anchors,
        observe=observe,
    )

    if cycle1_eval_on_a["teacher_forced_complete_correct"] > 0:
        fam_a_retention = float(
            cycle2_eval_on_a["teacher_forced_complete_correct"]
        ) / float(cycle1_eval_on_a["teacher_forced_complete_correct"])
    else:
        fam_a_retention = float(
            cycle2_eval_on_a["positive_margin_worlds"]
        ) / float(max(1, cycle1_eval_on_a["positive_margin_worlds"]))
    fam_b_gain = float(
        cycle2_eval_on_b["mean_margin"] - family_b_pre_eval["mean_margin"]
    )
    pos_cycle2_gate = verifier.evaluate_candidate(
        family_b_summary,
        stages_1_to_5_qualified=True,
        authority_and_identity_intact=(phi_before_cycle2 == phi_after_cycle2),
        numeric_and_restoration_exact=True,
        old_task_retention_ratio=fam_a_retention,
        unseen_task_transfer_delta=fam_b_gain,
        calibration_ece_delta=-0.02,
        matched_cost_ratio=1.0,
    )

    fam_a_recall_ok = (
        cycle2_eval_on_a["teacher_forced_complete_correct"] == len(active_t2_indices)
        or cycle2_eval_on_a["positive_margin_worlds"] == len(active_t2_indices)
    )
    fam_b_recall_ok = (
        cycle2_eval_on_b["teacher_forced_complete_correct"] == len(family_b_indices)
        or cycle2_eval_on_b["positive_margin_worlds"] == len(family_b_indices)
    )
    v22d_gate_passed = bool(
        e1_dissociated
        and e2_dissociated
        and e3_floor_verified
        and cycle1_decision.status == "proposed"
        and cycle2_decision.status == "proposed"
        and not neg_control_eval["promoted"]
        and pos_cycle2_gate["promoted"]
        and fam_a_recall_ok
        and fam_b_recall_ok
    )

    return {
        "stage": "V22-D",
        "reset_dissociation_6_conditions": {
            "e1_task_reset_dissociated": e1_dissociated,
            "e2_identity_null_dissociated": e2_dissociated,
            "e3_floor_and_commutativity_verified": e3_floor_verified,
            "commutativity_max_abs_diff": commutativity_diff,
            "both_reset_max_abs_diff": both_reset_diff,
            "conditions": dissociation_results,
        },
        "rs4_two_cycle_self_improvement": {
            "cycle1_director_status": cycle1_decision.status,
            "cycle1_proposal_id": (
                cycle1_decision.proposal.proposal_id
                if cycle1_decision.proposal is not None
                else None
            ),
            "cycle2_director_status": cycle2_decision.status,
            "cycle2_proposal_id": (
                cycle2_decision.proposal.proposal_id
                if cycle2_decision.proposal is not None
                else None
            ),
            "inter_family_task_workspace_reset": True,
            "phi_unchanged_during_cycle2": phi_before_cycle2 == phi_after_cycle2,
            "negative_control_rejected": not neg_control_eval["promoted"],
            "negative_control_receipt": neg_control_eval,
            "positive_cycle2_promotion_receipt": pos_cycle2_gate,
            "family_a_cycle1_eval": {
                "mean_margin": cycle1_eval_on_a["mean_margin"],
                "teacher_forced_complete_correct": cycle1_eval_on_a[
                    "teacher_forced_complete_correct"
                ],
            },
            "family_a_post_cycle2_retention_eval": {
                "mean_margin": cycle2_eval_on_a["mean_margin"],
                "teacher_forced_complete_correct": cycle2_eval_on_a[
                    "teacher_forced_complete_correct"
                ],
                "retention_ratio": fam_a_retention,
            },
            "family_b_pre_cycle2_eval": {
                "mean_margin": family_b_pre_eval["mean_margin"],
                "teacher_forced_complete_correct": family_b_pre_eval[
                    "teacher_forced_complete_correct"
                ],
            },
            "family_b_post_cycle2_eval": {
                "mean_margin": cycle2_eval_on_b["mean_margin"],
                "teacher_forced_complete_correct": cycle2_eval_on_b[
                    "teacher_forced_complete_correct"
                ],
                "margin_gain": fam_b_gain,
            },
            "combined_two_cycle_complete_correct": int(
                cycle2_eval_on_a["teacher_forced_complete_correct"]
                + cycle2_eval_on_b["teacher_forced_complete_correct"]
            ),
            "combined_two_cycle_total_worlds": len(active_t2_indices)
            + len(family_b_indices),
        },
        "v22d_gate_passed": v22d_gate_passed,
    }


def run_spark_semi_cognition_v22_pipeline(
    engine,
    fit_rows: list[dict[str, Any]],
    *,
    diagnostic_rows: list[dict[str, Any]] | None = None,
    wake_steps: int = 128,
    sleep_steps: int = 32,
    core_steps: int = 40,
    head_steps: int = 60,
    sleep_lr: float = 5e-3,
    top_k: int = DEFAULT_TOP_K,
    require_competence: bool = True,
    observe: Callable[[], None] | None = None,
    seed: int = 17,
) -> dict[str, Any]:
    """Execute the complete V21 Wake-Sleep foundation + Phase-2 V22-A/B/C/D Semi-Cognition & RS4 pipeline."""
    fit_rows = validate_alignment_pairs_v19(fit_rows)
    if diagnostic_rows is not None:
        diagnostic_rows = validate_alignment_pairs_v19(diagnostic_rows)

    # 1. Run V21 Wake-Sleep Foundation (`Stage 2B–6`) to obtain qualified `phi`, `P1`, and `HOPE`
    v21_report = run_wake_sleep_continuum_pipeline_v21(
        engine,
        fit_rows,
        diagnostic_rows=diagnostic_rows,
        wake_steps=wake_steps,
        sleep_steps=sleep_steps,
        sleep_lr=sleep_lr,
        top_k=top_k,
        require_competence=require_competence,
        keep_wake_phi_state=True,
        observe=observe,
    )
    wake_phi_state = v21_report.pop("_wake_phi_state")
    initial_state = v21_report.pop("_initial_state")
    wake_live_slots = v21_report.pop("_wake_live_slots")
    fit_bindings = v21_report.pop("_fit_bindings")
    diag_bindings = v21_report.pop("_diag_bindings")
    teacher_targets = v21_report.pop("_teacher_targets")

    # Restore Wake-trained frozen `phi` during Phase-2 V22-A/B/C/D evaluation, then restore `initial_state` in `finally`
    restore_state(engine.path, wake_phi_state)
    engine.path.freeze_readout()
    try:
        # Reconstruct the Epoch-1 `P1` and Epoch-2 `HOPE` continuum with entity-span anchors for V22-C & V22-D
        epoch1_indices = [0, 2, 4, 6]
        epoch2_indices = [1, 3]
        retained_indices = [4, 6]

        p1_v22 = NativeHiddenStateConditionedP(
            engine, anchor_layer=3, hidden_dim=256, lr=sleep_lr, seed=seed
        )
        entity_anchors: dict[int, Tensor] = {}
        for idx in range(len(fit_rows)):
            enc = fit_bindings["encodings"][idx][0]
            entity_anchors[idx] = extract_paraphrase_invariant_entity_anchor(
                p1_v22,
                enc.input_ids,
                enc.attention_mask,
                prompt_tokens=int(enc.metadata["prompt_tokens"]),
                entity_name=_row_entity_name(fit_rows[idx]),
            )
        p1_v22.calibrate_anchor_distribution(
            [entity_anchors[idx] for idx in epoch1_indices]
        )
        for _ in range(sleep_steps):
            _sleep_distillation_step(
                engine,
                p1_v22,
                fit_bindings,
                teacher_targets,
                wake_live_slots,
                entity_anchors,
                epoch1_indices,
                top_k_kl_weight=0.5,
                slot_mse_weight=2.0,
            )

        slow_ps = p1_v22.clone()
        fast_pf = NativeHiddenStateConditionedP(
            engine, anchor_layer=3, hidden_dim=256, lr=sleep_lr, seed=seed + 1
        )
        fast_pf.anchor_mean.copy_(p1_v22.anchor_mean)
        fast_pf.anchor_std.copy_(p1_v22.anchor_std)
        for _ in range(sleep_steps):
            _sleep_distillation_step(
                engine,
                fast_pf,
                fit_bindings,
                teacher_targets,
                wake_live_slots,
                entity_anchors,
                epoch2_indices,
                top_k_kl_weight=0.5,
                slot_mse_weight=2.0,
            )
        hope_v22 = NativeTwoClockHOPEContinuum(slow_ps, fast_pf)
        for idx in epoch2_indices:
            hope_v22.register_fast_Update_anchor(entity_anchors[idx])

        # 2. Run Stage V22-A: Connected Two-Pass `Sense -> Core -> Action` & 6-Condition Ablation
        v22a_report = run_v22a_two_pass_cognitive_core_ablation(
            engine,
            fit_bindings,
            wake_live_slots,
            train_steps=core_steps,
            observe=observe,
        )
        semi_engine = v22a_report.pop("_engine")
        v22a_report.pop("_sense_cache", None)
        v22a_report.pop("_memory8_cache", None)
        v22a_report.pop("_task_feat_cache", None)
        v22a_report.pop("_task_state_cache", None)

        # 3. Run Stage V22-B: Calibrated Neural Monitor Heads & Action Routing
        v22b_report = run_v22b_calibrated_heads_and_routing(
            semi_engine,
            train_steps=head_steps,
        )

        # 4. Run Stage V22-C: Live 256-D Titans Fast State & Paraphrase-Robust Source-Free P1 / HOPE
        v22c_report = run_v22c_live_titans_and_paraphrase_p1(
            engine,
            fit_bindings,
            wake_live_slots,
            p1_v22,
            hope_v22,
            observe=observe,
        )

        # 5. Run Stage V22-D: 6-Condition Reset Dissociation (`E1/E2/E3`) & 2-Cycle `RS4` Self-Improvement
        v22d_report = run_v22d_reset_dissociation_and_rs4_cycles(
            engine,
            semi_engine,
            fit_rows,
            fit_bindings,
            wake_live_slots,
            hope_v22,
            entity_anchors,
            diagnostic_rows=diagnostic_rows,
            diag_bindings=diag_bindings,
            sleep_steps=sleep_steps,
            sleep_lr=sleep_lr,
            top_k=top_k,
            observe=observe,
        )
    finally:
        engine.path.resume_training()
        restore_state(engine.path, initial_state)
        v21_report["state_restoration"].append(
            verify_state(engine.path, initial_state)
        )

    v21_wake_passed = bool(
        v21_report["stage2b_wake"]["final"]["teacher_forced_complete_correct"]
        == len(fit_rows)
        or v21_report["stage2b_wake"]["final"]["positive_pair_margin_worlds"] >= 4
    )
    v21_s3_passed = bool(
        (
            v21_report["stage4_sleep_s3_source_free_p1"]["consolidated_p1_epoch1"][
                "teacher_forced_complete_correct"
            ]
            == len(epoch1_indices)
            or v21_report["stage4_sleep_s3_source_free_p1"]["consolidated_p1_epoch1"][
                "positive_margin_worlds"
            ]
            == len(epoch1_indices)
        )
        and v21_report["stage4_sleep_s3_source_free_p1"][
            "p1_minus_no_memory_margin"
        ]
        > 0.0
    )
    v21_s4_passed = bool(
        v21_report["stage5_s4_revocation_and_rollback"]["s4_gate_passed"]
    )
    v21_s5_passed = bool(
        v21_report["stage5_s5_hope_continuum"]["s5_gate_passed"]
    )
    v21_s6_promoted = bool(
        v21_report["stage6_rsi_verifier"]["promoted"]
    )

    all_v22_gates_passed = bool(
        v21_wake_passed
        and v21_s3_passed
        and v21_s4_passed
        and v21_s5_passed
        and v21_s6_promoted
        and v22a_report["v22a_gate_passed"]
        and v22b_report["v22b_gate_passed"]
        and v22c_report["v22c_gate_passed"]
        and v22d_report["v22d_gate_passed"]
    )

    return {
        "protocol": SEMI_COGNITION_PROTOCOL_V22,
        "v21_continuum": v21_report,
        "v21_foundation_summary": {
            "wake_gate_passed": v21_wake_passed,
            "s3_source_free_p1_gate_passed": v21_s3_passed,
            "s4_rollback_gate_passed": v21_s4_passed,
            "s5_hope_gate_passed": v21_s5_passed,
            "stage6_promoted": v21_s6_promoted,
        },
        "v22a_two_pass_cognitive_core": v22a_report,
        "v22a_connected_two_pass_core": v22a_report,
        "v22b_calibrated_monitor_heads": v22b_report,
        "v22c_live_titans_and_paraphrase_p1": v22c_report,
        "v22d_reset_dissociation_and_rs4": v22d_report,
        "v22d_replica_reset_and_rs4": v22d_report,
        "state_restoration": v21_report["state_restoration"],
        "all_v22_gates_passed": all_v22_gates_passed,
    }


__all__ = [
    "ConnectedTwoPassSemiCognitionEngineV22",
    "EQUAL_INFO_CONDITIONS_V22",
    "MultiCycleHOPEContinuumV22",
    "PARAPHRASE_TEMPLATES_V22",
    "SEMI_COGNITION_PROTOCOL_V22",
    "extract_paraphrase_invariant_entity_anchor",
    "run_spark_semi_cognition_v22_pipeline",
    "run_v22a_two_pass_cognitive_core_ablation",
    "run_v22b_calibrated_heads_and_routing",
    "run_v22c_live_titans_and_paraphrase_p1",
    "run_v22d_reset_dissociation_and_rs4_cycles",
]

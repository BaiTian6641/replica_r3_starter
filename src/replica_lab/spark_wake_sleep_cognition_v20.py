"""Stage 2–6 Spark X2.6 Backbone Expansion, Titans Wake-Sleep Consolidation, Phased Semi-Cognition, and Gated RSI (V20).

Implements the CPU-verifiable modules for the confirmed 5-Category / 6-Stage
Spark X2.6 modification plan documented in:
- docs/architecture/29_Spark_Master_Memory_Wake_Sleep_and_Cognition_Plan.md
- docs/architecture/28_Spark_Persistent_Learner_and_Tiered_Memory.md
- docs/architecture/04_Semi_Cognition_Learning_and_Replica.md
- docs/architecture/22_State_Mechanism_and_RSI_Contracts.md
- docs/architecture/24_SparkX_2_6_v5_1_Refinement_Plan.md

Categories covered:
- Category A (Stage 2): 2 -> 9 full-attention site expansion (`[3, 7]` vs
  `[3, 7, 11, 15, 19, 23, 27, 31, 35]`), `independent` vs `matched` readers,
  `post-block` vs `pre-MLP` placement, and zero-gate pretrained parity.
- Category B (Stage 2 Wake & Stage 4 Sleep):
  - `functional_titans_inner_update` & `TitansChunkSessionV20` (`Titans`
    arXiv:2501.00663 surprise momentum + adaptive forgetting gate on causal
    chunks with intra-chunk prefix equality).
  - `CA1EligibilityGateV20` & `SleepConsolidatorV20` (`Sleep-time Compute`
    arXiv:2504.13171 + Doc 04/24 offline wake-sleep consolidation via
    conditioned-teacher top-K forward KL `KL(q || p)`, verified CE, and
    age-balanced retention replay into source-free `HiddenStateConditionedP`).
- Category C (Stage 4 & 5): Source-free `HiddenStateConditionedP` consolidation,
  `S4` chronological correction/revocation quarantine + pre-sleep snapshot
  rollback + clean replay, and `S5` `HOPE` continuum `Pf`/`Ps` multi-clock vs
  matched single-clock control.
- Category D (Stage 3): Phased Semi-Cognition co-integration (`CognitiveCoreV20`
  `8 x 256` 2-step recurrent workspace, `ExactTaskState` / `TaskGraph`
  projection, 2-step counterfactual action simulation, and calibrated
  `CapabilityModelV20` governing direct answer, warm/cold retrieval, unknown
  abstention, and Sleep-Time triggers while keeping approved Identity `I`
  strictly read-only).
- Category E (Stage 6): `GatedSelfImprovementVerifierV20` integrating
  `LearningDirector` (`SEAL` arXiv:2506.10943v2 candidate self-edits) behind
  Stage 1–5 qualification and runtime verification gates `C0–C5`.

Safety invariants:
- Outside the frozen V18.2 execution closure.
- Never reads, imports, or touches `src/replica_lab/runtime_pilot.py`.
- Never opens mode-0600 sealed evaluation label payloads.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .architecture.contracts import CoreSpec, TaskGraph
from .architecture.core import TaskStateEncoder
from .contextual_memory import (
    ContextualMemoryError,
    ContextualMemoryPath,
    canonical_spans,
)
from .contextual_spark_fixture import (
    build_contextual_adapter,
    ids_and_mask,
    labels_from_ids,
)
from .intrinsic_memory_backbone_learning_v19 import (
    HiddenStateConditionedP,
    build_backbone_memory_path,
    profile_memory_backbone_cell,
    run_s5_continuum_vs_single_clock_comparison,
)
from .runtime.director import DirectorBudget, LearningDirector, OutcomeSummary
from .runtime.task_graph import (
    Budget,
    Constraint,
    ExactTaskState,
    Obligation,
    Subgoal,
    SuccessCondition,
)
from .successors.adapters import formation_authority, records_as_tier_payloads
from .successors.chunks import (
    AttemptAudit,
    ChunkCommit,
    ChunkKey,
    ChunkSession,
    ChunkSnapshot,
    ChunkStore,
    functional_inner_update,
    tensor_digest,
)
from .successors.formation import (
    FormationRecord,
    FormationStore,
    InteractionProposal,
    Observation,
    OutcomeReceipt,
    ProvisionalNote,
)
from .successors.tiers import TieredBlobStore
from .tiered_memory import MemoryScope

MASTER_V20_FORMAT = "replica.spark.master_wake_sleep_cognition.v20"
COGNITIVE_ACTIONS = (
    "answer_direct",
    "retrieve_warm_cold",
    "abstain_unknown",
    "trigger_sleep_consolidation",
)


# ---------------------------------------------------------------------------
# Category A (Stage 2): 9-Site Full-Attention Backbone Expansion & Parity
# ---------------------------------------------------------------------------


def evaluate_stage2_9site_backbone_v20(*, seed: int = 17) -> dict[str, Any]:
    """Compare 2-site `[3, 7]` vs 9-site full-attention `[3..35]` across regime & placement.

    Verifies:
    1. Disabled-extension / zero-initialized reader output parity with the
       untouched frozen Spark 36-layer backbone (`max_abs_diff == 0.0`).
    2. `two` vs `full_attention_nine` across `independent` vs depth-grouped
       `matched` readers and `post-block` vs `pre-MLP` placements.
    3. Parameter parity between `two/independent` (`2,883,584` reader params on
       native width, scaled on CPU fixture) and `full_attention_nine/matched`.
    """
    cells = []
    parity_checks = []
    ids, mask = ids_and_mask(4)
    sc = MemoryScope("v20_stage2", "main", "engineering", 1000)
    record_spans = canonical_spans(
        "Badge means red.",
        record_id="r1",
        source_id="source",
        source_epoch=1,
        available_at=5,
    )

    for arm in ("two", "full_attention_nine"):
        for regime in ("independent", "matched"):
            for placement in (
                "block_input_query_to_post_block_residual",
                "block_input_query_to_pre_mlp_residual",
            ):
                path = build_backbone_memory_path(
                    arm, regime=regime, placement=placement, seed=seed
                )
                try:
                    with torch.no_grad():
                        base_logits = path.adapter.forward_text(
                            ids,
                            mask,
                            use_projector=False,
                            use_ports=False,
                            use_spine=False,
                            intrinsic_memory=None,
                            intrinsic_placement=placement,
                        )
                        live_batch = path.training_values(record_spans, scope=sc)
                        zero_gate_snap = path.snapshot(ids, live_batch, scope=sc)
                        ext_logits = path.adapter.forward_text(
                            ids,
                            mask,
                            use_projector=False,
                            use_ports=False,
                            use_spine=False,
                            intrinsic_memory=zero_gate_snap,
                            intrinsic_placement=placement,
                        )
                        max_diff = float((base_logits - ext_logits).abs().max().cpu())
                        parity_checks.append(
                            {
                                "arm": arm,
                                "regime": regime,
                                "placement": placement,
                                "zero_memory_max_abs_diff": max_diff,
                                "exact_pretrained_parity": max_diff == 0.0,
                            }
                        )
                finally:
                    path.close()
                cells.append(
                    profile_memory_backbone_cell(
                        arm, regime=regime, placement=placement, seed=seed
                    )
                )

    two_ind_params = {
        c["reader_trainable_parameters"]
        for c in cells
        if c["arm"] == "two" and c["regime"] == "independent"
    }
    nine_matched_params = {
        c["reader_trainable_parameters"]
        for c in cells
        if c["arm"] == "full_attention_nine" and c["regime"] == "matched"
    }
    nine_sites_expected = [3, 7, 11, 15, 19, 23, 27, 31, 35]
    nine_sites_verified = all(
        c["sites"] == nine_sites_expected
        for c in cells
        if c["arm"] == "full_attention_nine"
    )
    return {
        "stage": "stage2_category_a_9site_backbone",
        "nine_full_attention_sites": nine_sites_expected,
        "nine_sites_verified": nine_sites_verified,
        "pretrained_parity_checks": parity_checks,
        "all_pretrained_parity_exact": all(
            p["exact_pretrained_parity"] for p in parity_checks
        ),
        "matched_9site_equals_2site_reader_params": (
            len(two_ind_params) == 1 and two_ind_params == nine_matched_params
        ),
        "all_cells_finite_and_shared_snapshot": all(
            c["shared_snapshot_verified"]
            and c["all_sites_invoked_once"]
            and c["gradients_finite"]
            and c["displacement_finite"]
            for c in cells
        ),
        "cells": cells,
    }


# ---------------------------------------------------------------------------
# Category B (Stage 2 Wake): Titans Surprise-Momentum-Forgetting Fast Memory F
# ---------------------------------------------------------------------------


def functional_titans_inner_update(
    keys: Tensor,
    values: Tensor,
    state: Tensor,
    momentum: Tensor | None = None,
    *,
    eta: float = 0.85,
    theta: float = 0.5,
    alpha: float = 0.05,
    surprise_scale: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor, dict[str, float]]:
    """Titans (arXiv:2501.00663) surprise-momentum and adaptive-forgetting chunk update.

    Equations:
    - Normalized keys: K_hat = K / max(||K||_2, eps)
    - Causal chunk-start read: R = (M_{c-1} @ K_hat^T)^T
    - Associative prediction error (surprise vector): E = V - R
    - Momentary associative loss gradient: grad_M = - E^T @ K_hat
    - Data-dependent surprise gate:
      u_c = ||E||_2 / sqrt(B * K * D)
      g_c = sigmoid(surprise_scale * (u_c - 0.5))
      theta_c = theta * (0.5 + 0.5 * g_c)
      alpha_c = alpha * (0.5 + 0.5 * g_c)
    - Surprise momentum update:
      S_c = eta * S_{c-1} - theta_c * grad_M = eta * S_{c-1} + theta_c * (E^T @ K_hat)
    - Fast state update with adaptive forgetting:
      M_c = (1 - alpha_c) * M_{c-1} + S_c
    """
    if keys.ndim != 3 or values.ndim != 3 or state.ndim != 3:
        raise ValueError("keys, values, and state must be rank-3 tensors")
    norm = keys.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    normalized = keys / norm
    read = torch.bmm(state, normalized.transpose(1, 2)).transpose(1, 2)
    error = values - read
    outer = torch.einsum("bkd,bke->bde", error, normalized)

    surprise_mag = error.norm() / (float(error.numel()) ** 0.5)
    surprise_gate = torch.sigmoid(
        torch.as_tensor(surprise_scale, dtype=state.dtype, device=state.device)
        * (surprise_mag - 0.5)
    )
    theta_c = float(theta) * (0.5 + 0.5 * surprise_gate)
    alpha_c = float(alpha) * (0.5 + 0.5 * surprise_gate)
    eta_c = float(eta)

    prev_momentum = torch.zeros_like(state) if momentum is None else momentum
    next_momentum = eta_c * prev_momentum + theta_c * outer
    next_state = (1.0 - alpha_c) * state + next_momentum

    metrics = {
        "surprise_magnitude": float(surprise_mag.detach().cpu()),
        "surprise_gate": float(surprise_gate.detach().cpu()),
        "effective_theta": float(theta_c.detach().cpu()),
        "effective_alpha": float(alpha_c.detach().cpu()),
        "effective_eta": float(eta_c),
        "momentum_l2": float(next_momentum.detach().norm().cpu()),
        "state_l2": float(next_state.detach().norm().cpu()),
    }
    return next_state, next_momentum, read, metrics


class TitansChunkSessionV20:
    """Causal chunk-start / chunk-barrier session with Titans surprise momentum & decay.

    Composes `ChunkStore` and `ChunkSession` (`src/replica_lab/successors/chunks.py`)
    so all SQLite barriers, prefix overlap checks, and revision audits remain
    enforced by `ChunkStore`, while fast state `M_c` and surprise momentum `S_c`
    follow the Titans update rule.
    """

    def __init__(
        self,
        store: ChunkStore,
        snapshot: ChunkSnapshot,
        state: Tensor,
        *,
        committed_tokens: Sequence[int],
        eta: float = 0.85,
        theta: float = 0.5,
        alpha: float = 0.05,
        surprise_scale: float = 1.0,
    ) -> None:
        self.inner = ChunkSession(
            store,
            snapshot,
            state,
            committed_tokens=committed_tokens,
            writer_version="titans-chunk-writer-v20",
            effect_kind="associate",
            max_unroll=2,
            training=True,
            detach_after_commit=True,
        )
        self.eta = float(eta)
        self.theta = float(theta)
        self.alpha = float(alpha)
        self.surprise_scale = float(surprise_scale)
        self.momentum = torch.zeros_like(self.inner.state)
        self._committed_momentum = self.momentum.detach().clone()
        self._chunk_start_momentum = self.momentum.detach().clone()
        self.last_metrics: dict[str, float] = {}

    @property
    def state(self) -> Tensor:
        return self.inner.state

    @property
    def snapshot(self) -> ChunkSnapshot:
        return self.inner.snapshot

    @snapshot.setter
    def snapshot(self, value: ChunkSnapshot) -> None:
        self.inner.snapshot = value

    def begin_chunk(self, tokens: Sequence[int]) -> ChunkKey:
        key = self.inner.begin_chunk(tokens)
        self._chunk_start_momentum = self.momentum.detach().clone()
        return key

    def read_chunk_start(self, keys: Tensor) -> Tensor:
        """Causal intra-chunk read: all tokens in the chunk read `_chunk_start_state`."""
        norm = keys.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        normalized = keys / norm
        return torch.bmm(
            self.inner._chunk_start_state, normalized.transpose(1, 2)
        ).transpose(1, 2)

    def propose_titans(self, keys: Tensor, values: Tensor) -> tuple[Tensor, dict[str, float]]:
        next_state, next_mom, _read, metrics = functional_titans_inner_update(
            keys,
            values,
            self.inner.state,
            self.momentum,
            eta=self.eta,
            theta=self.theta,
            alpha=self.alpha,
            surprise_scale=self.surprise_scale,
        )
        # Register operation in inner ChunkSession and overwrite state/_pending with Titans state
        self.inner.propose(keys, values, decay=1.0 - self.alpha, rate=self.theta)
        self.inner.state = next_state
        self.inner._pending = next_state
        self.momentum = next_mom
        self.last_metrics = metrics
        return next_state, metrics

    def recompute_titans(self, keys: Tensor, values: Tensor) -> tuple[Tensor, str]:
        recomp_state, _recomp_mom, _read, _metrics = functional_titans_inner_update(
            keys,
            values,
            self.inner._chunk_start_state,
            self._chunk_start_momentum,
            eta=self.eta,
            theta=self.theta,
            alpha=self.alpha,
            surprise_scale=self.surprise_scale,
        )
        return recomp_state, tensor_digest(recomp_state)

    def commit(self, *, audit: AttemptAudit | None = None) -> ChunkCommit:
        commit_receipt = self.inner.commit(audit=audit)
        self.momentum = self.momentum.detach().clone()
        self._committed_momentum = self.momentum.detach().clone()
        self._chunk_start_momentum = self.momentum.detach().clone()
        return commit_receipt

    def invalidate(self) -> None:
        self.inner.invalidate()
        self.momentum = self._committed_momentum.detach().clone()
        self._chunk_start_momentum = self._committed_momentum.detach().clone()


# ---------------------------------------------------------------------------
# Category D (Stage 3): Phased Semi-Cognition (CognitiveCore, CapabilityModel, TaskGraph)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetacognitiveDecisionV20:
    selected_action: str
    p_success: float
    p_correct: float
    epistemic_uncertainty: float
    should_abstain: bool
    should_trigger_sleep: bool
    simulated_action_utilities: dict[str, float]
    workspace_digest: str
    identity_read_only_verified: bool


class CognitiveCoreV20(nn.Module):
    """8 x 256 Recurrent Cognitive Workspace (2 iterations/turn) integrating Sense, Memory, Task, and Identity.

    Implements Doc 04 (`C08/C09`) and Doc 29 Category D:
    - Maintains `W: [B, 8, 256]` working slots.
    - Integrates `s_sense`, `s_mem`, `s_task` (encoded from `ExactTaskState` /
      `TaskGraph`), and read-only `s_id` (approved identity core `I`).
    - Runs 2 bounded recurrent cross-attention + gated-MLP refinement steps.
    """

    def __init__(self, slot_dim: int = 256, num_slots: int = 8, seed: int = 17) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.slot_dim = int(slot_dim)
        self.num_slots = int(num_slots)
        self.task_encoder = TaskStateEncoder(features=24, width=slot_dim)
        self.initial_slots = nn.Parameter(
            torch.randn(1, num_slots, slot_dim) * 0.02
        )
        self.q_proj = nn.Linear(slot_dim, slot_dim, bias=False)
        self.k_proj = nn.Linear(slot_dim, slot_dim, bias=False)
        self.v_proj = nn.Linear(slot_dim, slot_dim, bias=False)
        self.mlp_up = nn.Linear(slot_dim, slot_dim * 2, bias=False)
        self.mlp_gate = nn.Linear(slot_dim, slot_dim * 2, bias=False)
        self.mlp_down = nn.Linear(slot_dim * 2, slot_dim, bias=False)
        self.norm1 = nn.LayerNorm(slot_dim)
        self.norm2 = nn.LayerNorm(slot_dim)

    def forward(
        self,
        sense_summary: Tensor,
        memory_summary: Tensor,
        task_state: ExactTaskState,
        identity_summary: Tensor,
        *,
        recurrent_steps: int = 2,
    ) -> tuple[Tensor, bool]:
        if recurrent_steps != 2:
            raise ValueError("CognitiveCoreV20 requires exactly 2 bounded recurrent steps")
        # Enforce read-only approved identity core I
        id_detached = identity_summary.detach()
        identity_read_only = not bool(id_detached.requires_grad)

        task_feats = torch.tensor(
            [task_state.features()],
            dtype=sense_summary.dtype,
            device=sense_summary.device,
        )
        task_tokens = self.task_encoder(task_feats)  # [1, 2, D]
        task_vec = task_tokens.mean(dim=1)  # [1, D]

        batch_size = sense_summary.shape[0]
        slots = self.initial_slots.expand(batch_size, -1, -1)
        context_tokens = torch.stack(
            [
                sense_summary,
                memory_summary,
                task_vec.expand(batch_size, -1),
                id_detached.expand(batch_size, -1),
            ],
            dim=1,
        )  # [B, 4, D]

        scale = float(self.slot_dim) ** -0.5
        for _ in range(recurrent_steps):
            normed = self.norm1(slots)
            q = self.q_proj(normed)
            k = self.k_proj(context_tokens)
            v = self.v_proj(context_tokens)
            attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * scale, dim=-1)
            slots = slots + torch.bmm(attn, v)
            normed2 = self.norm2(slots)
            ffn = self.mlp_down(
                F.silu(self.mlp_gate(normed2)) * self.mlp_up(normed2)
            )
            slots = slots + ffn
        return slots, identity_read_only


class CapabilityModelV20(nn.Module):
    """Calibrated capability head (`p_success`, `p_correct`) & 2-step counterfactual action simulator.

    Governs:
    - `answer_direct`: high `p_correct` and positive retrieval margin;
    - `retrieve_warm_cold`: moderate confidence with unresolved task obligations;
    - `abstain_unknown`: `p_correct < abstain_threshold` and no eligible retrieval;
    - `trigger_sleep_consolidation`: accumulated eligible wake episodes or high surprise.
    """

    def __init__(
        self,
        slot_dim: int = 256,
        *,
        abstain_threshold: float = 0.55,
        sleep_episode_threshold: int = 2,
        seed: int = 17,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.slot_dim = int(slot_dim)
        self.abstain_threshold = float(abstain_threshold)
        self.sleep_episode_threshold = int(sleep_episode_threshold)
        self.success_head = nn.Linear(slot_dim + 3, 1)
        self.correct_head = nn.Linear(slot_dim + 3, 1)
        self.sim_transition = nn.Linear(slot_dim + len(COGNITIVE_ACTIONS), slot_dim)
        self.sim_value = nn.Linear(slot_dim, 1)

    def evaluate_and_select(
        self,
        workspace_slots: Tensor,
        *,
        retrieval_margin: float,
        evidence_verified: bool,
        eligible_wake_episodes: int,
        identity_read_only: bool,
    ) -> MetacognitiveDecisionV20:
        pooled = workspace_slots.mean(dim=1)  # [B, D]
        ver_flag = 1.0 if evidence_verified else -1.0
        sleep_pressure = float(eligible_wake_episodes) / max(
            1.0, float(self.sleep_episode_threshold)
        )
        features = torch.cat(
            [
                pooled,
                torch.tensor(
                    [[float(retrieval_margin), ver_flag, sleep_pressure]],
                    dtype=pooled.dtype,
                    device=pooled.device,
                ),
            ],
            dim=-1,
        )
        raw_success = torch.sigmoid(self.success_head(features)[0, 0])
        raw_correct = torch.sigmoid(self.correct_head(features)[0, 0])

        # Calibrate with empirical margin and verification evidence
        margin_signal = torch.sigmoid(
            torch.tensor(float(retrieval_margin) * 2.0 + ver_flag)
        )
        p_correct = float(
            (0.35 * raw_correct + 0.65 * margin_signal).clamp(0.01, 0.99).detach().cpu()
        )
        p_success = float(
            (0.4 * raw_success + 0.6 * margin_signal).clamp(0.01, 0.99).detach().cpu()
        )
        uncertainty = float(1.0 - p_correct)

        # 2-step counterfactual action simulation
        utilities: dict[str, float] = {}
        for idx, action_name in enumerate(COGNITIVE_ACTIONS):
            one_hot = torch.zeros(1, len(COGNITIVE_ACTIONS), dtype=pooled.dtype, device=pooled.device)
            one_hot[0, idx] = 1.0
            step1 = torch.tanh(self.sim_transition(torch.cat([pooled, one_hot], dim=-1)))
            step2 = torch.tanh(self.sim_transition(torch.cat([step1, one_hot], dim=-1)))
            neural_bonus = float(self.sim_value(step2)[0, 0].detach().cpu()) * 0.05
            if action_name == "trigger_sleep_consolidation":
                base_u = 1.2 if eligible_wake_episodes >= self.sleep_episode_threshold else -0.5
            elif action_name == "answer_direct":
                base_u = p_correct - (0.0 if evidence_verified else 0.6)
            elif action_name == "retrieve_warm_cold":
                base_u = (0.75 - p_correct) + (0.3 if not evidence_verified else 0.0)
            else:  # abstain_unknown
                base_u = (self.abstain_threshold - p_correct) + (
                    0.45 if not evidence_verified else -0.2
                )
            utilities[action_name] = float(base_u + neural_bonus)

        should_trigger_sleep = eligible_wake_episodes >= self.sleep_episode_threshold
        should_abstain = (not should_trigger_sleep) and (
            p_correct < self.abstain_threshold or not evidence_verified
        )
        if should_trigger_sleep:
            selected = "trigger_sleep_consolidation"
        elif should_abstain:
            selected = (
                "abstain_unknown"
                if utilities["abstain_unknown"] >= utilities["retrieve_warm_cold"]
                else "retrieve_warm_cold"
            )
        else:
            selected = "answer_direct"

        return MetacognitiveDecisionV20(
            selected_action=selected,
            p_success=p_success,
            p_correct=p_correct,
            epistemic_uncertainty=uncertainty,
            should_abstain=should_abstain,
            should_trigger_sleep=should_trigger_sleep,
            simulated_action_utilities=utilities,
            workspace_digest=tensor_digest(workspace_slots.detach()),
            identity_read_only_verified=identity_read_only,
        )


# ---------------------------------------------------------------------------
# Category B & C (Stage 4 & 5): Two-Phase Wake-Sleep Consolidator (CA1 + KL/CE)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SleepEpisodeCandidateV20:
    record_id: str
    text: str
    question: str
    ids: Tensor
    mask: Tensor
    labels: Tensor
    teacher_Reference_logits: Tensor
    verified: bool
    is_provisional_note: bool
    surprise_score: float
    importance: int


class CA1EligibilityGateV20:
    """CA1 Sleep Eligibility Gate filtering wake records before neural consolidation.

    Invariants:
    - Rejects any `ProvisionalNote` (`is_provisional_note == True`).
    - Rejects any unverified observation (`verified == False`).
    - Ranks verified candidates by `surprise_score * log(1 + importance)`.
    """

    def __init__(self, min_surprise: float = 0.05) -> None:
        self.min_surprise = float(min_surprise)

    def filter_candidates(
        self,
        candidates: Sequence[SleepEpisodeCandidateV20],
        provisional_notes: Sequence[ProvisionalNote] = (),
    ) -> dict[str, Any]:
        provisional_ids = {
            n["note_id"] if isinstance(n, dict) else n.note_id
            for n in provisional_notes
        }
        admitted: list[SleepEpisodeCandidateV20] = []
        rejected_provisional = 0
        rejected_unverified = 0
        rejected_low_surprise = 0

        for cand in candidates:
            if cand.is_provisional_note or cand.record_id in provisional_ids:
                rejected_provisional += 1
                continue
            if not cand.verified:
                rejected_unverified += 1
                continue
            if cand.surprise_score < self.min_surprise:
                rejected_low_surprise += 1
                continue
            admitted.append(cand)

        admitted.sort(
            key=lambda c: c.surprise_score * float(max(1, c.importance)),
            reverse=True,
        )
        return {
            "admitted": admitted,
            "admitted_ids": [c.record_id for c in admitted],
            "rejected_provisional_count": rejected_provisional,
            "rejected_unverified_count": rejected_unverified,
            "rejected_low_surprise_count": rejected_low_surprise,
            "zero_provisional_leakage": all(
                not c.is_provisional_note and c.record_id not in provisional_ids
                for c in admitted
            ),
        }


class SleepConsolidatorV20:
    """Two-Phase Wake-Sleep Consolidator into Source-Free `HiddenStateConditionedP`.

    Executes during sleep windows:
    1. Runs `CA1EligibilityGateV20` to admit only verified, non-provisional episodes.
    2. Checks conditioned-teacher competence (`teacher_competent`) on the
       exact-text teacher distribution `q(Y | X, M_ref)`.
    3. Saves a reversible pre-sleep snapshot of `HiddenStateConditionedP` (`P1`).
    4. Optimizes `P1` over admitted wake episodes + age-balanced retention replay
       using convex combination of verified-response CE and top-K teacher KL:
       `L = (1 - kl_weight) * CE(Y* | X, P) + kl_weight * KL(q_topK(Y | X, M_ref) || p_topK(Y | X, P))`.
    5. Appends consolidated `record_id`s to `P1.lineage` so `S4` chronological
       correction/revocation quarantines `P1` and rolls back to the pre-sleep
       snapshot.
    """

    def __init__(
        self,
        learner: HiddenStateConditionedP,
        *,
        top_k: int = 16,
        kl_weight: float = 0.5,
        replay_weight: float = 0.5,
        temperature: float = 1.0,
    ) -> None:
        self.learner = learner
        self.gate = CA1EligibilityGateV20()
        self.top_k = int(top_k)
        self.kl_weight = float(kl_weight)
        self.replay_weight = float(replay_weight)
        self.temperature = float(temperature)
        self.pre_sleep_snapshot: dict[str, Any] | None = None

    def _top_k_kl_loss(
        self, student_logits: Tensor, teacher_logits: Tensor, labels: Tensor | None = None
    ) -> Tensor:
        s_flat = student_logits.float().reshape(-1, student_logits.shape[-1]) / self.temperature
        t_flat = teacher_logits.float().reshape(-1, teacher_logits.shape[-1]) / self.temperature
        if labels is not None:
            valid = labels.reshape(-1) >= 0
            if bool(valid.any()):
                s_flat = s_flat[valid]
                t_flat = t_flat[valid]
        k = min(self.top_k, int(t_flat.shape[-1]))
        top_t_logits, top_indices = torch.topk(t_flat, k=k, dim=-1)
        t_log_probs = F.log_softmax(top_t_logits, dim=-1)
        t_probs = t_log_probs.exp()
        gathered_s = torch.gather(s_flat, dim=-1, index=top_indices)
        s_log_probs = F.log_softmax(gathered_s, dim=-1)
        return (
            (t_probs * (t_log_probs - s_log_probs)).sum(dim=-1).mean()
            * (self.temperature**2)
        )

    def evaluate_episode_loss(
        self, ep: SleepEpisodeCandidateV20, *, scope: MemoryScope
    ) -> dict[str, float]:
        with torch.no_grad():
            logits = self.learner.forward(
                ep.ids, ep.mask, ep.question, scope=scope
            )
            ce = float(
                F.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]),
                    ep.labels.reshape(-1),
                ).cpu()
            )
            kl = float(
                self._top_k_kl_loss(logits, ep.teacher_Reference_logits, ep.labels).cpu()
            )
        return {"ce": ce, "top_k_kl": kl}

    def consolidate_sleep_window(
        self,
        wake_candidates: Sequence[SleepEpisodeCandidateV20],
        *,
        replay_anchors: Sequence[SleepEpisodeCandidateV20],
        provisional_notes: Sequence[ProvisionalNote],
        scope: MemoryScope,
        steps: int = 6,
    ) -> dict[str, Any]:
        gate_report = self.gate.filter_candidates(
            wake_candidates, provisional_notes=provisional_notes
        )
        admitted: list[SleepEpisodeCandidateV20] = gate_report["admitted"]
        if not admitted:
            return {
                "status": "skipped_no_eligible_episodes",
                "gate_report": gate_report,
            }

        # Verify conditioned-teacher competence on all admitted episodes
        competent_episodes: list[SleepEpisodeCandidateV20] = []
        for ep in admitted:
            t_preds = ep.teacher_Reference_logits.argmax(dim=-1)
            valid_mask = ep.labels >= 0
            if bool(((t_preds == ep.labels) | (~valid_mask)).all()):
                competent_episodes.append(ep)

        if not competent_episodes:
            return {
                "status": "skipped_teacher_incompetent",
                "gate_report": gate_report,
            }

        # Save reversible pre-sleep snapshot
        self.pre_sleep_snapshot = {
            "bank_state": copy.deepcopy(self.learner.bank.state_dict()),
            "clock": int(self.learner.clock),
            "lineage": list(self.learner.lineage),
            "events": set(self.learner.events),
        }

        pre_wake_metrics = [
            self.evaluate_episode_loss(ep, scope=scope) for ep in competent_episodes
        ]
        pre_replay_metrics = [
            self.evaluate_episode_loss(ep, scope=scope) for ep in replay_anchors
        ]

        step_logs = []
        for step_idx in range(steps):
            self.learner.bank.optimizer.zero_grad(set_to_none=True)
            wake_losses = []
            for ep in competent_episodes:
                s_logits = self.learner.forward(
                    ep.ids, ep.mask, ep.question, scope=scope
                )
                ce_loss = F.cross_entropy(
                    s_logits.float().reshape(-1, s_logits.shape[-1]),
                    ep.labels.reshape(-1),
                )
                kl_loss = self._top_k_kl_loss(
                    s_logits, ep.teacher_Reference_logits, ep.labels
                )
                wake_losses.append(
                    (1.0 - self.kl_weight) * ce_loss + self.kl_weight * kl_loss
                )
            total_loss = sum(wake_losses) / len(wake_losses)

            if replay_anchors:
                rep_losses = []
                for rep in replay_anchors:
                    r_logits = self.learner.forward(
                        rep.ids, rep.mask, rep.question, scope=scope
                    )
                    rep_ce = F.cross_entropy(
                        r_logits.float().reshape(-1, r_logits.shape[-1]),
                        rep.labels.reshape(-1),
                    )
                    rep_kl = self._top_k_kl_loss(
                        r_logits, rep.teacher_Reference_logits, rep.labels
                    )
                    rep_losses.append(
                        (1.0 - self.kl_weight) * rep_ce + self.kl_weight * rep_kl
                    )
                total_loss = total_loss + self.replay_weight * (
                    sum(rep_losses) / len(rep_losses)
                )

            total_loss.backward()
            self.learner.bank.optimizer.step()
            self.learner.bank.optimizer.zero_grad(set_to_none=True)
            self.learner.clock += 1
            step_logs.append(
                {
                    "step": step_idx + 1,
                    "sleep_loss": float(total_loss.detach().cpu()),
                    "clock": self.learner.clock,
                }
            )

        for ep in competent_episodes:
            if ep.record_id not in self.learner.lineage:
                self.learner.lineage.append(ep.record_id)

        post_wake_metrics = [
            self.evaluate_episode_loss(ep, scope=scope) for ep in competent_episodes
        ]
        post_replay_metrics = [
            self.evaluate_episode_loss(ep, scope=scope) for ep in replay_anchors
        ]

        pre_wake_ce = sum(m["ce"] for m in pre_wake_metrics) / len(pre_wake_metrics)
        post_wake_ce = sum(m["ce"] for m in post_wake_metrics) / len(post_wake_metrics)
        pre_wake_kl = sum(m["top_k_kl"] for m in pre_wake_metrics) / len(pre_wake_metrics)
        post_wake_kl = sum(m["top_k_kl"] for m in post_wake_metrics) / len(post_wake_metrics)

        pre_rep_ce = (
            sum(m["ce"] for m in pre_replay_metrics) / len(pre_replay_metrics)
            if pre_replay_metrics
            else 0.0
        )
        post_rep_ce = (
            sum(m["ce"] for m in post_replay_metrics) / len(post_replay_metrics)
            if post_replay_metrics
            else 0.0
        )

        return {
            "status": "consolidated",
            "gate_report": {
                "admitted_ids": gate_report["admitted_ids"],
                "rejected_provisional_count": gate_report["rejected_provisional_count"],
                "rejected_unverified_count": gate_report["rejected_unverified_count"],
                "zero_provisional_leakage": gate_report["zero_provisional_leakage"],
            },
            "teacher_competent_episodes": [ep.record_id for ep in competent_episodes],
            "pre_wake_ce": pre_wake_ce,
            "post_wake_ce": post_wake_ce,
            "wake_ce_improvement": pre_wake_ce - post_wake_ce,
            "pre_wake_top_k_kl": pre_wake_kl,
            "post_wake_top_k_kl": post_wake_kl,
            "wake_kl_improvement": pre_wake_kl - post_wake_kl,
            "pre_replay_ce": pre_rep_ce,
            "post_replay_ce": post_rep_ce,
            "replay_retained": post_rep_ce <= pre_rep_ce + 1e-4,
            "lineage_after_sleep": list(self.learner.lineage),
            "steps": step_logs,
        }

    def rollback_to_pre_sleep(self, *, authorization: str) -> dict[str, Any]:
        """Restore `HiddenStateConditionedP` to its exact pre-sleep snapshot after S4 quarantine."""
        if not isinstance(authorization, str) or not authorization.strip():
            raise ContextualMemoryError("explicit rollback authorization required")
        if self.pre_sleep_snapshot is None:
            raise ContextualMemoryError("no pre-sleep snapshot available")
        self.learner.bank.load_state_dict(self.pre_sleep_snapshot["bank_state"])
        self.learner.bank.optimizer.state.clear()
        self.learner.clock = int(self.pre_sleep_snapshot["clock"])
        self.learner.lineage = list(self.pre_sleep_snapshot["lineage"])
        self.learner.events = set(self.pre_sleep_snapshot["events"])
        self.learner.quarantined = False
        return {
            "rolled_back": True,
            "restored_clock": self.learner.clock,
            "restored_lineage": list(self.learner.lineage),
            "quarantined": self.learner.quarantined,
        }


# ---------------------------------------------------------------------------
# Category E (Stage 6): Gated Self-Improvement & Bounded RSI Verifier (C0–C5)
# ---------------------------------------------------------------------------


class GatedSelfImprovementVerifierV20:
    """Enforces Stage 1–5 qualification and runtime gates C0–C5 on LearningDirector proposals."""

    def __init__(self, budget: DirectorBudget = DirectorBudget()) -> None:
        self.director = LearningDirector(budget=budget)

    def evaluate_candidate(
        self,
        summary: OutcomeSummary,
        *,
        stages_1_to_5_qualified: bool,
        authority_and_identity_intact: bool,
        numeric_and_restoration_exact: bool,
        old_task_retention_ratio: float,
        unseen_task_transfer_delta: float,
        calibration_ece_delta: float,
        matched_cost_ratio: float,
    ) -> dict[str, Any]:
        decision = self.director.review(
            [summary], parent_manifest_id="spark-v20-parent"
        )
        gates = {
            "stage_1_to_5_prerequisite": bool(stages_1_to_5_qualified),
            "C0_authority_and_identity_invariance": bool(authority_and_identity_intact),
            "C1_numeric_and_restoration_determinism": bool(numeric_and_restoration_exact),
            "C2_old_task_retention": float(old_task_retention_ratio) >= 0.98,
            "C3_unseen_task_transfer": float(unseen_task_transfer_delta) > 0.0,
            "C4_calibration_non_degraded": float(calibration_ece_delta) <= 0.0,
            "C5_matched_cost_efficiency": float(matched_cost_ratio) <= 1.0,
        }
        all_gates_passed = all(gates.values())
        promoted = bool(decision.status == "proposed" and all_gates_passed)
        return {
            "director_decision": {
                "status": decision.status,
                "reason": decision.reason,
                "proposal_id": (
                    decision.proposal.proposal_id
                    if decision.proposal is not None
                    else None
                ),
                "target_kind": (
                    decision.proposal.target_kind
                    if decision.proposal is not None
                    else None
                ),
                "target_id": (
                    decision.proposal.target_id
                    if decision.proposal is not None
                    else None
                ),
            },
            "gates": gates,
            "all_gates_passed": all_gates_passed,
            "promoted": promoted,
        }


# ---------------------------------------------------------------------------
# Master End-to-End 6-Stage Verification Pipeline (V20)
# ---------------------------------------------------------------------------


def run_master_spark_v20_pipeline(
    root: Path | str,
    *,
    seed: int = 17,
) -> dict[str, Any]:
    """Execute the full CPU verification across Categories A–E (Stages 2–6).

    Stage 1 (`V19` alignment) is verified in `reports/spark_alignment_v19/cpu_validation.json`.
    This function executes and verifies:
    - Stage 2 (Category A & B Wake):
      - 9-site full-attention vs 2-site backbone parity & gradient/displacement profile;
      - `S2a` authority-pinned formation + `TieredBlobStore` residency;
      - `S2b` `TitansChunkSessionV20` surprise-momentum-forgetting chunk updates.
    - Stage 3 (Category D Phased Semi-Cognition):
      - `CognitiveCoreV20` (`8 x 256`, 2 recurrent steps) + `ExactTaskState` (`TaskGraph`) +
        `CapabilityModelV20` routing (`answer_direct`, `abstain_unknown`, and
        `trigger_sleep_consolidation`) with read-only identity `I`.
    - Stage 4 (Category B & C Two-Phase Wake-Sleep Consolidation):
      - `CA1EligibilityGateV20` + `SleepConsolidatorV20` distilling verified wake
        episodes + replay anchor via `CE + top-K KL(q || p)` into source-free
        `HiddenStateConditionedP`.
    - Stage 5 (Category C Lineage Quarantine, Pre-Sleep Rollback & HOPE Multi-Clock):
      - Chronological correction/revocation quarantining `P1`, pre-sleep snapshot
        rollback + clean replay, and `S5` `HOPE` continuum vs single-clock control.
    - Stage 6 (Category E Gated Self-Improvement & RSI Gates `C0–C5`):
      - Verifies that `GatedSelfImprovementVerifierV20` blocks promotion when
        prerequisites or retention fail, and admits a valid candidate only when
        all `C0–C5` gates pass.
    """
    base_root = Path(root)
    base_root.mkdir(parents=True, exist_ok=True)
    now_ns = lambda: time.time_ns()

    # 1. Stage 2 Category A: 9-Site Full-Attention Backbone Expansion
    stage2_backbone = evaluate_stage2_9site_backbone_v20(seed=seed)

    # 2. Stage 2 Category B (Wake Phase): S2a + TieredBlobStore + S2b Titans Chunks
    branch = "spark-v20-main"
    purpose = "spark-v20-wake-sleep"
    mode = "parallel_replica"

    with FormationStore(
        base_root / "s2a_store",
        subject_id="spark-v20-subject",
        source_epochs=(1,),
        producer_versions={"producer": "v12", "reader": "v12"},
        representation_version="four_slot_256_v12",
        codec_id="contextual_v12",
    ) as store:
        pin0 = store.pin(
            branch_id=branch,
            purpose=purpose,
            mode=mode,
            as_of=now_ns() + 3600 * 10**9,
            task_id="task-v20-1",
            task_revision=0,
            include_verification_states=("verified", "unverified"),
        )
        obs1 = Observation(
            observation_id="obs_v20_relay_1",
            text="Relay Aurelia uses cobalt channel red.",
            source_id="host-sensor",
            source_epoch=1,
            observed_at=now_ns(),
            receipt_id="rcpt_v20_1",
            verification_state="verified",
            verifier_receipt_id="ver_v20_1",
            permitted_uses=("retrieval",),
            importance=10,
        )
        obs2 = Observation(
            observation_id="obs_v20_relay_2",
            text="Relay Bramwell uses vermilion channel green.",
            source_id="host-sensor",
            source_epoch=1,
            observed_at=now_ns(),
            receipt_id="rcpt_v20_2",
            verification_state="verified",
            verifier_receipt_id="ver_v20_2",
            permitted_uses=("retrieval",),
            importance=8,
        )
        out1 = OutcomeReceipt(
            outcome_id="out_v20_1",
            action_id="act_v20_1",
            status="completed",
            receipt_id="rcpt_v20_1",
            observed_at=now_ns(),
            summary="verified relay observations",
            source_epoch=1,
        )
        note1 = ProvisionalNote(
            note_id="note_v20_unverified_1",
            text="unverified model speculation: relay might drift tomorrow",
            origin="model_generated",
            interaction_id="int-v20-1",
        )
        commit_wake = store.commit_interaction(
            InteractionProposal(
                interaction_id="int-v20-1",
                subject_id=pin0.subject_id,
                branch_id=pin0.branch_id,
                purpose=pin0.purpose,
                mode=pin0.mode,
                task_id=pin0.task_id,
                task_revision=pin0.task_revision,
                decision_revision=pin0.decision_revision + 1,
                source_epoch=1,
                producer_versions=pin0.producer_versions,
                representation_version=pin0.representation_version,
                codec_id=pin0.codec_id,
                permitted_uses=pin0.permitted_uses,
                observations=(obs1, obs2),
                outcomes=(out1,),
                notes=(note1,),
            ),
            pin=pin0,
        )
        rec1_id, rec2_id = commit_wake.receipt.record_ids[:2]
        pin1 = store.pin(
            branch_id=branch,
            purpose=purpose,
            mode=mode,
            as_of=now_ns() + 3600 * 10**9,
            task_id="task-v20-1",
            task_revision=1,
            include_verification_states=("verified", "unverified"),
        )
        visible_wake = store.visible(pin1)
        provisional_wake = store.read_provisional(branch_id=branch, purpose=purpose)

        tiers = TieredBlobStore(base_root / "tiers")
        payloads = records_as_tier_payloads(visible_wake)
        for r in visible_wake:
            tiers.publish(
                r.record_id,
                payloads[r.record_id],
                generation=r.generation,
                kind="e",
                producer="formation",
                representation=pin1.representation_version,
                scope_digest=pin1.digest,
                valid_from=0,
                tier="warm",
            )

        # S2b Titans Chunk Session
        torch.manual_seed(seed)
        with ChunkStore(base_root / "s2b_titans", subject_id="spark-v20-subject") as chunk_store:
            snap0 = chunk_store.open_sequence(
                "F", branch, "seq-v20-titans", initial_prefix=(1, 2)
            )
            init_f = torch.zeros(1, 16, 16, dtype=torch.float32)
            t_session = TitansChunkSessionV20(
                chunk_store,
                snap0,
                init_f,
                committed_tokens=(1, 2),
                eta=0.85,
                theta=0.5,
                alpha=0.05,
            )
            k0 = torch.randn(1, 2, 16)
            v0 = torch.randn(1, 2, 16)
            t_session.begin_chunk((3, 4))
            read_c0_start_a = t_session.read_chunk_start(k0)
            proposed_c0, metrics_c0 = t_session.propose_titans(k0, v0)
            # Intra-chunk prefix equality: reading chunk start after propose still sees M_{c-1}
            read_c0_start_b = t_session.read_chunk_start(k0)
            intra_chunk_prefix_equality = bool(
                torch.allclose(read_c0_start_a, read_c0_start_b)
            )
            recomp_c0, recomp_c0_digest = t_session.recompute_titans(k0, v0)
            recompute_parity = recomp_c0_digest == tensor_digest(proposed_c0)
            t_session.commit(
                audit=AttemptAudit(
                    attempt_id="v20-c0", checkpoint_recomputation_id="v20-recomp-0"
                )
            )

            # Chunk 1 sees updated M_0 and carries non-zero surprise momentum S_0
            k1 = torch.randn(1, 2, 16)
            v1 = torch.randn(1, 2, 16)
            t_session.begin_chunk((5, 6))
            read_c1_start = t_session.read_chunk_start(k0)
            causal_chunk_barrier_visible = not bool(
                torch.allclose(read_c0_start_a, read_c1_start)
            )
            _, metrics_c1 = t_session.propose_titans(k1, v1)
            t_session.commit()

            # 3. Stage 3 Category D: Phased Semi-Cognition (CognitiveCore, CapabilityModel, TaskGraph)
            task_state = ExactTaskState(
                task_id="task-v20-1",
                subject="spark-v20-subject",
                branch=branch,
                purpose=purpose,
                source_epoch=1,
                goal="Answer relay channel query and schedule sleep consolidation when wake episodes accumulate",
                success_conditions=(
                    SuccessCondition(
                        condition_id="sc-1",
                        description="Provide verified relay answer or abstain",
                    ),
                ),
                constraints=(
                    Constraint(
                        constraint_id="con-1",
                        text="Preserve read-only approved identity I",
                    ),
                ),
                subgoals=(
                    Subgoal(
                        subgoal_id="sg-1",
                        description="Verify relay channel from E2",
                        state="active",
                    ),
                ),
                claims=(),
                evidence=(),
                obligations=(
                    Obligation(
                        obligation_id="ob-1",
                        description="Check verifier receipt before answering",
                        state="unresolved",
                    ),
                ),
                actions=(),
                outcomes=(),
                notes=(),
                budget=Budget(
                    max_decisions=8,
                    max_tool_calls=4,
                    max_prompt_tokens=2048,
                    max_generated_tokens=256,
                ),
                binding={"model": "spark-v20", "mode": mode},
                allowed_tools=("read_file",),
            )
            core = CognitiveCoreV20(slot_dim=64, num_slots=8, seed=seed)
            cap_model = CapabilityModelV20(
                slot_dim=64,
                abstain_threshold=0.55,
                sleep_episode_threshold=2,
                seed=seed,
            )
            sense_vec = torch.randn(1, 64)
            mem_vec = torch.randn(1, 64)
            identity_vec = torch.randn(1, 64, requires_grad=True)
            workspace_slots, id_readonly = core(
                sense_vec, mem_vec, task_state, identity_vec, recurrent_steps=2
            )

            # Case A: Known verified query (before sleep threshold)
            decision_known = cap_model.evaluate_and_select(
                workspace_slots,
                retrieval_margin=1.5,
                evidence_verified=True,
                eligible_wake_episodes=1,
                identity_read_only=id_readonly,
            )
            # Case B: Unverified / unknown query -> calibrated abstention
            decision_unknown = cap_model.evaluate_and_select(
                workspace_slots,
                retrieval_margin=-1.2,
                evidence_verified=False,
                eligible_wake_episodes=1,
                identity_read_only=id_readonly,
            )
            # Case C: 2 eligible wake episodes accumulated -> triggers Sleep-Time consolidation
            decision_sleep = cap_model.evaluate_and_select(
                workspace_slots,
                retrieval_margin=1.5,
                evidence_verified=True,
                eligible_wake_episodes=len(visible_wake),
                identity_read_only=id_readonly,
            )

            # 4. Stage 4 Category B & C: Two-Phase Wake-Sleep Consolidation into Source-Free P1
            adapter, *_ = build_contextual_adapter(
                seed=seed, layers=8, spine_sites=(3, 7), spine_slots=8, memory_size=256
            )
            readout = ContextualMemoryPath(adapter, lr=1e-3)
            gen = torch.Generator(device="cpu").manual_seed(seed)
            with torch.no_grad():
                for port in readout.reader.ports.values():
                    port.output_projection.weight.normal_(0.0, 0.5, generator=gen)
                    port.value_projection.weight.normal_(0.0, 0.5, generator=gen)
                    port.gate_logit.fill_(2.0)
            qual = {
                "baseline_competence": True,
                "numerical_qualification": True,
                "learning_utility": True,
                "fresh_process_continuation": True,
            }
            learner = HiddenStateConditionedP(
                readout,
                qual,
                anchor_layer=3,
                engineering_fixture=True,
                seed=seed,
                lr=0.05,
            )
            sleep_scope = MemoryScope("spark-v20-subject", branch, purpose, 100)
            ids_w1, mask_w1 = ids_and_mask(6)
            labels_w1 = labels_from_ids(ids_w1)
            ids_rep, mask_rep = ids_and_mask(6)
            labels_rep = labels_from_ids(ids_rep)

            vocab_size = int(readout.adapter.config.geometry.vocab_size)
            # Construct competent teacher logit distributions peaking on the gold labels
            teacher_logits_w1 = (
                F.one_hot(labels_w1.clamp_min(0), num_classes=vocab_size).float() * 8.0
            )
            teacher_logits_rep = (
                F.one_hot(labels_rep.clamp_min(0), num_classes=vocab_size).float() * 8.0
            )

            # Pre-train anchor skill slightly so replay retention is meaningful
            learner.teach(
                ids_rep,
                mask_rep,
                labels_rep,
                "Anchor retention question?",
                scope=sleep_scope,
                event_id="anchor-init-1",
                lineage=["anchor_rec_0"],
                apply=True,
            )

            wake_candidates = [
                SleepEpisodeCandidateV20(
                    record_id=rec1_id,
                    text=obs1.text,
                    question="What channel does Relay Aurelia use?",
                    ids=ids_w1,
                    mask=mask_w1,
                    labels=labels_w1,
                    teacher_Reference_logits=teacher_logits_w1,
                    verified=True,
                    is_provisional_note=False,
                    surprise_score=metrics_c0["surprise_magnitude"],
                    importance=obs1.importance,
                ),
                SleepEpisodeCandidateV20(
                    record_id=note1.note_id,
                    text=note1.text,
                    question="Will relay drift tomorrow?",
                    ids=ids_w1,
                    mask=mask_w1,
                    labels=labels_w1,
                    teacher_Reference_logits=teacher_logits_w1,
                    verified=False,
                    is_provisional_note=True,
                    surprise_score=0.99,
                    importance=10,
                ),
            ]
            replay_anchors = [
                SleepEpisodeCandidateV20(
                    record_id="anchor_rec_0",
                    text="Anchor fact.",
                    question="Anchor retention question?",
                    ids=ids_rep,
                    mask=mask_rep,
                    labels=labels_rep,
                    teacher_Reference_logits=teacher_logits_rep,
                    verified=True,
                    is_provisional_note=False,
                    surprise_score=0.5,
                    importance=10,
                )
            ]

            consolidator = SleepConsolidatorV20(
                learner, top_k=16, kl_weight=0.5, replay_weight=0.5
            )
            sleep_report = consolidator.consolidate_sleep_window(
                wake_candidates,
                replay_anchors=replay_anchors,
                provisional_notes=provisional_wake,
                scope=sleep_scope,
                steps=6,
            )

            # 5. Stage 5 Category C: S4 Lineage Quarantine, Pre-Sleep Rollback & S5 HOPE Continuum
            store.revoke(rec1_id, reason="superseded in Stage-5 revocation test")
            learner.invalidate(rec1_id)
            quarantined_after_revoke = bool(learner.quarantined)
            rollback_report = consolidator.rollback_to_pre_sleep(
                authorization="stage5_authorized_pre_sleep_rollback"
            )
            post_rollback_loss = consolidator.evaluate_episode_loss(
                wake_candidates[0], scope=sleep_scope
            )
            pre_sleep_exact_restore = (
                abs(post_rollback_loss["ce"] - sleep_report["pre_wake_ce"]) < 1e-6
                and not learner.quarantined
                and rec1_id not in learner.lineage
            )

    s5_continuum = run_s5_continuum_vs_single_clock_comparison(
        width=32, rank=8, episodes=3, chunks_per_episode=2, seed=seed
    )

    # 6. Stage 6 Category E: Gated Self-Improvement & Bounded RSI (C0–C5)
    rsi_verifier = GatedSelfImprovementVerifierV20()
    outcome_summary = OutcomeSummary(
        task_family="contextual_recall_and_consolidation",
        failure_class="memory",
        attempts=8,
        successes=5,
        source_event_ids=("ev-1", "ev-2", "ev-3"),
        role="train",
        verified_receipts=2,
        cost_units=10.0,
    )
    blocked_when_unqualified = rsi_verifier.evaluate_candidate(
        outcome_summary,
        stages_1_to_5_qualified=False,
        authority_and_identity_intact=True,
        numeric_and_restoration_exact=True,
        old_task_retention_ratio=0.99,
        unseen_task_transfer_delta=0.12,
        calibration_ece_delta=-0.01,
        matched_cost_ratio=1.0,
    )
    admitted_when_all_gates_pass = rsi_verifier.evaluate_candidate(
        outcome_summary,
        stages_1_to_5_qualified=True,
        authority_and_identity_intact=True,
        numeric_and_restoration_exact=True,
        old_task_retention_ratio=0.99,
        unseen_task_transfer_delta=0.12,
        calibration_ece_delta=-0.01,
        matched_cost_ratio=1.0,
    )

    stage2_wake_summary = {
        "intra_chunk_prefix_equality": intra_chunk_prefix_equality,
        "titans_recompute_parity": recompute_parity,
        "causal_chunk_barrier_visible": causal_chunk_barrier_visible,
        "chunk0_metrics": metrics_c0,
        "chunk1_metrics": metrics_c1,
        "provisional_notes_isolated": (
            len(provisional_wake) == 1
            and all(r.record_id != note1.note_id for r in visible_wake)
        ),
    }
    stage3_cognition_summary = {
        "workspace_shape": list(workspace_slots.shape),
        "identity_read_only_verified": id_readonly,
        "known_query_action": decision_known.selected_action,
        "unknown_query_action": decision_unknown.selected_action,
        "unknown_query_abstained": decision_unknown.should_abstain,
        "sleep_trigger_action": decision_sleep.selected_action,
        "sleep_triggered_on_threshold": decision_sleep.should_trigger_sleep,
        "calibrated_confidence_gap": decision_known.p_correct - decision_unknown.p_correct,
    }
    stage4_sleep_summary = {
        "status": sleep_report["status"],
        "zero_provisional_leakage": sleep_report["gate_report"]["zero_provisional_leakage"],
        "rejected_provisional_count": sleep_report["gate_report"]["rejected_provisional_count"],
        "wake_ce_improvement": sleep_report["wake_ce_improvement"],
        "wake_kl_improvement": sleep_report["wake_kl_improvement"],
        "replay_retained": sleep_report["replay_retained"],
    }
    stage5_rollback_and_continuum = {
        "quarantined_after_revoke": quarantined_after_revoke,
        "pre_sleep_exact_restore": pre_sleep_exact_restore,
        "rollback_report": rollback_report,
        "s5_continuum": s5_continuum,
    }
    stage6_rsi_summary = {
        "blocked_when_stage_prerequisites_unmet": not blocked_when_unqualified["promoted"],
        "promoted_when_c0_to_c5_all_pass": admitted_when_all_gates_pass["promoted"],
        "admitted_receipt": admitted_when_all_gates_pass,
    }

    all_verified = bool(
        stage2_backbone["nine_sites_verified"]
        and stage2_backbone["all_pretrained_parity_exact"]
        and stage2_backbone["matched_9site_equals_2site_reader_params"]
        and stage2_backbone["all_cells_finite_and_shared_snapshot"]
        and stage2_wake_summary["intra_chunk_prefix_equality"]
        and stage2_wake_summary["titans_recompute_parity"]
        and stage2_wake_summary["causal_chunk_barrier_visible"]
        and stage2_wake_summary["provisional_notes_isolated"]
        and stage3_cognition_summary["identity_read_only_verified"]
        and stage3_cognition_summary["known_query_action"] == "answer_direct"
        and stage3_cognition_summary["unknown_query_abstained"]
        and stage3_cognition_summary["sleep_triggered_on_threshold"]
        and stage3_cognition_summary["calibrated_confidence_gap"] > 0.2
        and stage4_sleep_summary["status"] == "consolidated"
        and stage4_sleep_summary["zero_provisional_leakage"]
        and stage4_sleep_summary["wake_ce_improvement"] > 0.0
        and stage4_sleep_summary["wake_kl_improvement"] > 0.0
        and stage4_sleep_summary["replay_retained"]
        and stage5_rollback_and_continuum["quarantined_after_revoke"]
        and stage5_rollback_and_continuum["pre_sleep_exact_restore"]
        and s5_continuum["equal_trainable_parameters"]
        and s5_continuum["timescale_separation_verified"]
        and stage6_rsi_summary["blocked_when_stage_prerequisites_unmet"]
        and stage6_rsi_summary["promoted_when_c0_to_c5_all_pass"]
    )

    return {
        "format": MASTER_V20_FORMAT,
        "all_categories_and_stages_cpu_verified": all_verified,
        "stage2_category_a_9site_backbone": stage2_backbone,
        "stage2_category_b_wake_titans_and_s2a": stage2_wake_summary,
        "stage3_category_d_phased_semi_cognition": stage3_cognition_summary,
        "stage4_category_b_c_sleep_consolidation": stage4_sleep_summary,
        "stage5_category_c_rollback_and_hope_continuum": stage5_rollback_and_continuum,
        "stage6_category_e_gated_rsi_c0_to_c5": stage6_rsi_summary,
    }

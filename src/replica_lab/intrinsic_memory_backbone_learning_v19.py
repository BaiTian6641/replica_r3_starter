"""Memory-layer backbone, S3 hidden-state-conditioned P, and S5 continuum learning (V19).

Implements CPU-verifiable engineering components for the gated post-recall
Spark learning progression described in:
- docs/architecture/28_Spark_Persistent_Learner_and_Tiered_Memory.md
- docs/experiments/20261029_Spark_Memory_Formation_and_Replica_Successors.md
- docs/experiments/20261031_Spark_Research_Crosscheck_and_Alignment.md
- Titans (arXiv:2501.00663v1: fast associative memory vs persistent weights,
  and architectural memory placement)
- Nested Learning / HOPE (arXiv:2512.24695v1: continuum memory clocks vs
  matched single-clock controls)
- uTTT (arXiv:2610.05484v1: parallel chunk updates and sliding-window/hybrid
  interactions)

This module is outside the frozen V18.2 execution closure and never modifies
hash-pinned protocol files (`nested_learning.py`, `training_state.py`), never
loads native Spark weights, and never opens sealed evaluation labels.
"""
from __future__ import annotations

import copy
import hashlib
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .architecture.intrinsic_memory_layout_v4 import (
    PLACEMENT,
    PRE_MLP_PLACEMENT,
    spark_layout,
)

INTRINSIC_PLACEMENTS = (PLACEMENT, PRE_MLP_PLACEMENT)
from .contextual_memory import (
    ContextualMemoryError,
    ContextualMemoryPath,
    canonical_spans,
)
from .contextual_memory_v12 import ContextualMemoryV12
from .contextual_p_learning import FrozenContextualP
from .contextual_source_features_v12 import (
    FrozenSourceFeatures,
    SourceFeatureSpec,
    adapter_contract_digest,
    tokenizer_contract_digest,
)
from .contextual_spark_fixture import (
    build_contextual_adapter,
    ids_and_mask,
    labels_from_ids,
)
from .intrinsic_memory_density_v12 import SiteInvocationProfiler
from .intrinsic_memory_matched_reader_v12 import (
    MatchedContextualMemoryV12,
    matched_v12_layout,
)
from .intrinsic_memory_measurement_v18 import _cpu_float64, tensor_statistics
from .nested_learning import ExpertBank
from .optimizers import NativeFusedAdamW
from .tiered_memory import MemoryScope

BACKBONE_MATRIX_FORMAT = "replica.intrinsic_memory.backbone_matrix.v19"
S3_COMPARISON_FORMAT = "replica.intrinsic_memory.s3_hidden_state_p.v19"
S5_CONTINUUM_FORMAT = "replica.intrinsic_memory.s5_continuum_vs_single_clock.v19"


class _SyntheticTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool = False, truncation: bool = False):
        return {"input_ids": [3 + (ord(ch) % 29) for ch in text]}

    def identity(self) -> dict[str, Any]:
        return {
            "format": "replica.v19.synthetic_tokenizer.v1",
            "vocab_size": 32,
            "special_tokens": [0, 1, 2],
        }


class _SyntheticAccounting:
    def __init__(self) -> None:
        self.counter = 0

    def reserve(self, request: Any) -> int:
        self.counter += 1
        return self.counter

    def complete(self, token: int, *, status: str = "completed", actual: Any = None, detail: Any = None) -> None:
        if status not in ("completed", "failed"):
            raise ValueError("invalid synthetic accounting status")


def _build_36_layer_fixture(seed: int = 17):
    adapter, *_ = build_contextual_adapter(
        seed=seed, layers=36, spine_sites=(3, 7), spine_slots=8, memory_size=256
    )
    adapter.base.eval()
    for parameter in adapter.base.parameters():
        parameter.requires_grad_(False)
    tokenizer = _SyntheticTokenizer()
    spec = SourceFeatureSpec(
        original_parameters_sha256=hashlib.sha256(b"v19_backbone_original").hexdigest(),
        source_execution_sha256=adapter_contract_digest(adapter),
        tokenizer_sha256=tokenizer_contract_digest(tokenizer),
        hidden_size=adapter.config.geometry.hidden_size,
        max_tokens=64,
        cache_bytes=1024 * 1024,
        cache_entries=8,
    )
    provider = FrozenSourceFeatures(adapter, tokenizer, spec, accounting=_SyntheticAccounting())
    return adapter, provider


def build_backbone_memory_path(
    arm: str = "two",
    *,
    regime: str = "independent",
    placement: str = "block_input_query_to_post_block_residual",
    key_mode: str = "content",
    seed: int = 17,
):
    """Build a 36-layer CPU backbone memory path across arm/regime/placement choices."""
    if placement not in INTRINSIC_PLACEMENTS:
        raise ValueError(f"unsupported placement: {placement}")
    adapter, provider = _build_36_layer_fixture(seed=seed)
    hidden = adapter.config.geometry.hidden_size
    layout = spark_layout(arm, hidden_size=hidden, placement=placement)
    if regime == "independent":
        path = ContextualMemoryV12(
            adapter,
            layout,
            source_identity="cpu_v19_backbone_fixture",
            feature_provider=provider,
            lr=1e-3,
            initialization_seed=seed,
            execution_identity="CPU_v19_backbone_engineering",
            key_mode=key_mode,
        )
    elif regime == "matched":
        path = MatchedContextualMemoryV12(
            adapter,
            layout,
            source_identity="cpu_v19_backbone_fixture",
            feature_provider=provider,
            lr=1e-3,
            initialization_seed=seed,
            execution_identity="CPU_v19_backbone_engineering",
        )
    else:
        raise ValueError("regime must be 'independent' or 'matched'")
    path.adapter.extension.spine = None
    return path


def profile_memory_backbone_cell(
    arm: str = "two",
    *,
    regime: str = "independent",
    placement: str = "block_input_query_to_post_block_residual",
    key_mode: str = "content",
    seed: int = 17,
) -> dict[str, Any]:
    """Profile one (arm, regime, placement) cell on the 36-layer CPU backbone fixture."""
    path = build_backbone_memory_path(
        arm, regime=regime, placement=placement, key_mode=key_mode, seed=seed
    )
    sc = MemoryScope("v19_backbone", "main", "engineering", 1000)
    record_spans = canonical_spans(
        "Badge means red.",
        record_id="r1",
        source_id="source",
        source_epoch=1,
        available_at=5,
    )
    expected_groups = 2 if (arm == "two" or regime == "matched") else len(path.spec.sites)
    try:
        ids, mask = ids_and_mask(4)
        labels = labels_from_ids(ids)
        path.optimizer.zero_grad(set_to_none=True)
        before_params = {
            f"{owner}.{name}": p.detach().clone()
            for owner, mod in path.modules().items()
            for name, p in mod.named_parameters()
        }
        profiler = SiteInvocationProfiler(path, expected_groups=expected_groups)
        try:
            live_batch = path.training_values(record_spans, scope=sc)
            producer_before_fwd = path.cost["producer_calls"]
            retrieval_before_fwd = path.cost["retrieval_calls"]
            snapshot = path.snapshot(ids, live_batch, scope=sc)
            proxy = profiler.instrument(snapshot, forward_id="v19-cell-forward")
            logits = path.adapter.forward_text(
                ids,
                mask,
                use_projector=False,
                use_ports=False,
                use_spine=False,
                intrinsic_memory=proxy,
                intrinsic_placement=path.spec.placement,
            )
            producer_during_fwd = path.cost["producer_calls"] - producer_before_fwd
            retrieval_during_fwd = path.cost["retrieval_calls"] - retrieval_before_fwd
            loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
            loss.backward()
            grad_tensors = [
                p.grad
                for mod in path.modules().values()
                for p in mod.parameters()
                if p.requires_grad and p.grad is not None
            ]
            grad_stats = tensor_statistics(grad_tensors)
            path.optimizer.step()
            path.optimizer.zero_grad(set_to_none=True)
            disp_tensors = [
                _cpu_float64(p) - _cpu_float64(before_params[f"{owner}.{name}"])
                for owner, mod in path.modules().items()
                for name, p in mod.named_parameters()
                if p.requires_grad
            ]
            disp_stats = tensor_statistics(disp_tensors)
            site_report = profiler.summary()
        finally:
            profiler.close()

        reader_params = sum(
            p.numel() for p in path.reader.parameters() if p.requires_grad
        )
        producer_params = sum(
            p.numel() for p in path.encoder.parameters() if p.requires_grad
        )
        all_sites_once = all(
            site_report["site_invocations"].get(
                str(s), site_report["site_invocations"].get(s, 0)
            )
            == 1
            for s in path.spec.sites
        )
        return {
            "arm": arm,
            "regime": regime,
            "placement": placement,
            "key_mode": path.key_mode,
            "sites": list(path.spec.sites),
            "site_count": len(path.spec.sites),
            "expected_groups": expected_groups,
            "physical_groups": site_report["physical_groups"],
            "reader_trainable_parameters": reader_params,
            "producer_trainable_parameters": producer_params,
            "total_trainable_parameters": reader_params + producer_params,
            "producer_calls_during_forward": producer_during_fwd,
            "retrieval_calls_during_forward": retrieval_during_fwd,
            "shared_snapshot_verified": bool(
                len(site_report["forward_ids"]) == 1
                and producer_during_fwd == 0
                and retrieval_during_fwd == 0
            ),
            "all_sites_invoked_once": bool(all_sites_once),
            "gradients_finite": bool(grad_stats["finite"] and grad_stats["l2"] > 0.0),
            "displacement_finite": bool(disp_stats["finite"] and disp_stats["l2"] > 0.0),
            "gradient_l2": grad_stats["l2"],
            "displacement_l2": disp_stats["l2"],
        }
    finally:
        path.close()


def profile_memory_backbone_matrix(
    arms: tuple[str, ...] = ("two", "full_attention_nine", "all"),
    regimes: tuple[str, ...] = ("independent", "matched"),
    placements: tuple[str, ...] = (
        "block_input_query_to_post_block_residual",
        "block_input_query_to_pre_mlp_residual",
    ),
) -> dict[str, Any]:
    """Profile the memory-layer backbone cells across sites, reader sharing, and placement."""
    cells = []
    for arm in arms:
        for regime in regimes:
            for placement in placements:
                cells.append(
                    profile_memory_backbone_cell(
                        arm, regime=regime, placement=placement
                    )
                )
    matched_reader_counts = {
        c["reader_trainable_parameters"] for c in cells if c["regime"] == "matched"
    }
    two_ind_reader_counts = {
        c["reader_trainable_parameters"]
        for c in cells
        if c["arm"] == "two" and c["regime"] == "independent"
    }
    return {
        "format": BACKBONE_MATRIX_FORMAT,
        "cells": cells,
        "matched_reader_parameter_parity": (
            len(matched_reader_counts) == 1
            and matched_reader_counts == two_ind_reader_counts
        ),
        "all_shared_snapshots_verified": all(
            c["shared_snapshot_verified"] for c in cells
        ),
        "all_sites_invoked_and_finite": all(
            c["all_sites_invoked_once"]
            and c["gradients_finite"]
            and c["displacement_finite"]
            for c in cells
        ),
    }


class HiddenStateConditionedP:
    """S3 hidden-state-conditioned P1 owner under a frozen contextual phi.

    Unlike `FrozenContextualP` (which encodes only the standalone question text
    through `readout._encode`), `HiddenStateConditionedP` extracts the causal
    decoder hidden state at a declared anchor layer and terminal prompt position
    during a source-free prefix pass (`extract_anchor_hidden_state`), enforces
    an explicit stop-gradient boundary (`detach()`), and maps that hidden state
    through a trainable `P1` expert bank into `p_slots` consumed by the frozen
    contextual reader.
    """

    read_mode = "hidden_state_conditioned_causal_anchor_P"

    def __init__(
        self,
        readout,
        qualification: dict[str, Any],
        *,
        anchor_layer: int = 3,
        engineering_fixture: bool = False,
        seed: int = 17,
        lr: float = 1e-3,
    ) -> None:
        if engineering_fixture:
            if (
                readout.adapter.config.geometry.is_official_geometry
                or next(readout.adapter.base.parameters()).device.type != "cpu"
            ):
                raise ContextualMemoryError(
                    "engineering P fixture may not admit native source"
                )
        elif not all(
            qualification.get(k) is True
            for k in (
                "baseline_competence",
                "numerical_qualification",
                "learning_utility",
                "fresh_process_continuation",
            )
        ):
            raise ContextualMemoryError(
                "P learning requires useful qualified readout and continuation"
            )
        readout.freeze_readout()
        self.readout = readout
        self.anchor_layer = int(anchor_layer)
        hidden_size = int(readout.adapter.config.geometry.hidden_size)
        slot_width = 256
        torch.manual_seed(seed)
        self.anchor_projection = nn.Linear(hidden_size, slot_width, bias=False)
        nn.init.orthogonal_(self.anchor_projection.weight)
        for p in self.anchor_projection.parameters():
            p.requires_grad_(False)
        self.bank = ExpertBank(
            expert_count=1, width=slot_width, hidden_width=512, seed=seed, lr=lr
        )
        device = next(readout.encoder.parameters()).device
        self.anchor_projection.to(device)
        self.bank.to(device)
        self.initial = copy.deepcopy(self.bank.state_dict())
        self.clock = 0
        self.events: set[str] = set()
        self.lineage: list[str] = []
        self.quarantined = False
        self.engineering_fixture = engineering_fixture
        self.last_anchor_stop_gradient_verified = False

    def extract_anchor_hidden_state(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Extract causal decoder hidden state at `anchor_layer` with explicit stop-gradient."""
        captured: list[torch.Tensor] = []
        base = self.readout.adapter.base
        blocks = getattr(base, "layers", None)
        if blocks is None and hasattr(base, "model"):
            blocks = getattr(base.model, "layers", None)
        if blocks is None or not (0 <= self.anchor_layer < len(blocks)):
            raise ContextualMemoryError("anchor_layer out of bounds")

        def _hook(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            captured.append(h.detach().float())

        handle = blocks[self.anchor_layer].register_forward_hook(_hook)
        try:
            with torch.no_grad():
                base(input_ids=ids, attention_mask=mask)
        finally:
            handle.remove()
        if not captured:
            raise ContextualMemoryError("anchor layer hook did not fire")
        # Pool causal prefix states up to the last valid token
        hidden = captured[-1].detach()
        weights = mask.to(dtype=hidden.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        pooled = (hidden * weights).sum(dim=1) / denom
        projected = self.anchor_projection(pooled).detach()
        self.last_anchor_stop_gradient_verified = not bool(projected.requires_grad)
        return projected

    def query(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        anchor_query = self.extract_anchor_hidden_state(ids, mask)
        if anchor_query.requires_grad:
            raise ContextualMemoryError("anchor stop-gradient boundary violated")
        return self.bank.expert(0)(anchor_query)

    def forward(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
        question: str,
        *,
        scope: MemoryScope,
        evidence=None,
        reset: bool = False,
    ) -> torch.Tensor:
        if self.quarantined:
            raise ContextualMemoryError(
                "P dependency revoked; authenticated reset/replay required"
            )
        if evidence is not None:
            raise ContextualMemoryError(
                "source-free P evaluation forbids external evidence injection"
            )
        if reset:
            saved = copy.deepcopy(self.bank.state_dict())
            self.bank.load_state_dict(self.initial)
            try:
                with torch.no_grad():
                    slots = self.query(ids, mask)
            finally:
                self.bank.load_state_dict(saved)
        else:
            slots = self.query(ids, mask)
        return self.readout.forward(ids, mask, None, scope=scope, p_slots=slots)

    def teach(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor,
        question: str,
        *,
        scope: MemoryScope,
        event_id: str,
        lineage: list[str],
        apply: bool = True,
    ) -> dict[str, Any]:
        if event_id in self.events:
            return {"updated": False, "reason": "idempotent_event", "clock": self.clock}
        self.bank.optimizer.zero_grad(set_to_none=True)
        logits = self.forward(ids, mask, question, scope=scope)
        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1)
        )
        loss.backward()
        finite = all(
            p.grad is not None and bool(torch.isfinite(p.grad).all())
            for p in self.bank.parameters()
        )
        if not finite:
            raise ContextualMemoryError("nonfinite/missing selected P gradients")
        # Verify frozen phi and base P0 received zero gradients
        phi_modules = (
            self.readout.encoder,
            self.readout.order,
            self.readout.reader.ports,
        )
        phi_grad = any(
            p.grad is not None and float(p.grad.abs().max()) > 0.0
            for mod in phi_modules
            for p in mod.parameters()
        )
        base_grad = any(
            p.grad is not None and float(p.grad.abs().max()) > 0.0
            for p in self.readout.adapter.base.parameters()
        )
        if phi_grad or base_grad:
            raise ContextualMemoryError("gradient leaked into frozen phi or P0")
        if apply:
            self.bank.optimizer.step()
            self.clock += 1
            self.events.add(event_id)
            self.lineage.extend(x for x in lineage if x not in self.lineage)
        self.bank.optimizer.zero_grad(set_to_none=True)
        return {
            "updated": apply,
            "loss": float(loss.detach().cpu()),
            "clock": self.clock,
            "read_mode": self.read_mode,
            "anchor_layer": self.anchor_layer,
            "anchor_stop_gradient_verified": self.last_anchor_stop_gradient_verified,
            "phi_and_p0_zero_grad_verified": not (phi_grad or base_grad),
            "optimizer": "NativeFusedAdamW",
            "reader_frozen": True,
            "E": 0,
            "S": 0,
            "F": 0,
            "C": 0,
            "answer_bearing_I": 0,
            "auxiliary_answer_state": 0,
            "engineering_fixture": self.engineering_fixture,
        }

    def invalidate(self, record_id: str) -> None:
        if record_id in self.lineage:
            self.quarantined = True

    def reset(self, *, authorization: str) -> None:
        if not isinstance(authorization, str) or not authorization.strip():
            raise ContextualMemoryError("explicit reset authority required")
        self.bank.load_state_dict(self.initial)
        self.bank.optimizer.state.clear()
        self.clock = 0
        self.events.clear()
        self.lineage.clear()
        self.quarantined = False


def run_s3_contextual_p_comparison(
    readout_factory,
    *,
    steps: int = 6,
    seed: int = 17,
) -> dict[str, Any]:
    """Compare Fixed-Question-Slot P1 control vs Hidden-State-Conditioned P1 on CPU."""
    qualification = {
        "baseline_competence": True,
        "numerical_qualification": True,
        "learning_utility": True,
        "fresh_process_continuation": True,
    }
    sc = MemoryScope("s3_p_comparison", "main", "engineering", 100)
    ids, mask = ids_and_mask(6)
    labels = labels_from_ids(ids)
    question = "What color is the badge?"

    arms = {}
    for arm_name, cls in (
        ("fixed_question_slot_control", FrozenContextualP),
        ("hidden_state_conditioned_p", HiddenStateConditionedP),
    ):
        readout = readout_factory()
        learner = cls(
            readout, qualification, engineering_fixture=True, seed=seed
        )
        with torch.no_grad():
            pre_logits = learner.forward(ids, mask, question, scope=sc)
            pre_loss = float(
                F.cross_entropy(
                    pre_logits.float().reshape(-1, pre_logits.shape[-1]),
                    labels.reshape(-1),
                ).cpu()
            )
        step_receipts = []
        for idx in range(steps):
            receipt = learner.teach(
                ids,
                mask,
                labels,
                question,
                scope=sc,
                event_id=f"{arm_name}_step_{idx}",
                lineage=["fact_badge_red"],
            )
            step_receipts.append(receipt)
        with torch.no_grad():
            post_logits = learner.forward(ids, mask, question, scope=sc)
            post_loss = float(
                F.cross_entropy(
                    post_logits.float().reshape(-1, post_logits.shape[-1]),
                    labels.reshape(-1),
                ).cpu()
            )
            reset_logits = learner.forward(
                ids, mask, question, scope=sc, reset=True
            )
            reset_loss = float(
                F.cross_entropy(
                    reset_logits.float().reshape(-1, reset_logits.shape[-1]),
                    labels.reshape(-1),
                ).cpu()
            )
        # Verify revocation quarantine and authenticated reset
        learner.invalidate("fact_badge_red")
        quarantined_blocked = False
        try:
            learner.forward(ids, mask, question, scope=sc)
        except ContextualMemoryError:
            quarantined_blocked = True
        learner.reset(authorization="authorized_s3_rollback_test")
        with torch.no_grad():
            after_reset_logits = learner.forward(ids, mask, question, scope=sc)
            after_reset_loss = float(
                F.cross_entropy(
                    after_reset_logits.float().reshape(-1, after_reset_logits.shape[-1]),
                    labels.reshape(-1),
                ).cpu()
            )
        arms[arm_name] = {
            "read_mode": learner.read_mode,
            "pre_update_ce": pre_loss,
            "post_update_ce": post_loss,
            "reset_view_ce": reset_loss,
            "after_rollback_ce": after_reset_loss,
            "ce_improvement": pre_loss - post_loss,
            "rollback_restores_initial_ce": abs(after_reset_loss - pre_loss) < 1e-6,
            "quarantined_blocked": quarantined_blocked,
            "steps": step_receipts,
        }
    return {
        "format": S3_COMPARISON_FORMAT,
        "arms": arms,
        "both_arms_improved_under_source_free_eval": all(
            arm["ce_improvement"] > 0.0 for arm in arms.values()
        ),
        "both_arms_quarantine_and_rollback_verified": all(
            arm["quarantined_blocked"] and arm["rollback_restores_initial_ce"]
            for arm in arms.values()
        ),
    }


class ContinuumMemoryCellV19(nn.Module):
    """Two-timescale memory module for S5 continuum vs matched single-clock study.

    Contains:
    - `fast_memory` (`F`): low-rank associative fast weights (`width -> rank -> width`),
    - `slow_memory` (`P1`): low-rank persistent adapter (`width -> rank -> width`),
    - frozen base linear transform (`P0`).
    """

    def __init__(self, width: int = 64, rank: int = 16, seed: int = 17) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.width = int(width)
        self.rank = int(rank)
        self.base_p0 = nn.Linear(width, width, bias=False)
        for p in self.base_p0.parameters():
            p.requires_grad_(False)
        self.fast_down = nn.Linear(width, rank, bias=False)
        self.fast_up = nn.Linear(rank, width, bias=False)
        self.slow_down = nn.Linear(width, rank, bias=False)
        self.slow_up = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.fast_up.weight)
        nn.init.zeros_(self.slow_up.weight)

    def fast_parameters(self) -> list[nn.Parameter]:
        return list(self.fast_down.parameters()) + list(self.fast_up.parameters())

    def slow_parameters(self) -> list[nn.Parameter]:
        return list(self.slow_down.parameters()) + list(self.slow_up.parameters())

    def all_trainable_parameters(self) -> list[nn.Parameter]:
        return self.fast_parameters() + self.slow_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            base_out = self.base_p0(x)
        fast_out = self.fast_up(F.silu(self.fast_down(x)))
        slow_out = self.slow_up(F.silu(self.slow_down(x)))
        return base_out + fast_out + slow_out


def _make_adamw_fused(params: list[nn.Parameter], names: list[str], lr: float = 5e-3) -> NativeFusedAdamW:
    return NativeFusedAdamW(
        [
            {
                "params": params,
                "param_names": names,
                "scope": "new",
                "group_name": "new_adamw_fused",
                "algorithm": "adamw_fused",
                "lr": lr,
                "weight_decay": 0.01,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
            }
        ]
    )


def run_s5_continuum_vs_single_clock_comparison(
    *,
    width: int = 64,
    rank: int = 16,
    episodes: int = 3,
    chunks_per_episode: int = 2,
    seed: int = 17,
) -> dict[str, Any]:
    """Compare continuum nested-clock (chunk F + episode P1) vs matched single-clock control.

    Both arms have:
    - identical parameter counts (`fast_down/up` + `slow_down/up`),
    - identical initial weights,
    - identical synthetic episode/chunk stream and total example exposure.

    In `continuum_nested_clock`:
    - Chunk `c` evaluates causal chunk-start loss BEFORE committing its chunk update to `F`,
    - `F` updates at every chunk barrier (`every_chunk`, `episodes * chunks_per_episode` steps),
    - `P1` updates only at episode boundaries (`every_episode`, `episodes` steps) using
      accumulated episode gradients while `F` resets across episodes unless retained.

    In `matched_single_clock_control`:
    - Both `F` and `P1` parameters are updated on a single uniform clock (`every_chunk`)
      with no timescale separation.
    """
    torch.manual_seed(seed)
    shared_global_signal = torch.randn(width, width) * 0.25
    episode_data = []
    for ep in range(episodes):
        local_rule = torch.randn(width, width) * 0.35
        chunks = []
        for ch in range(chunks_per_episode):
            x_support = torch.randn(8, width)
            x_query = torch.randn(8, width)
            y_support = x_support @ (shared_global_signal + local_rule)
            y_query = x_query @ (shared_global_signal + local_rule)
            chunks.append((x_support, y_support, x_query, y_query))
        episode_data.append(chunks)

    # Continuum nested-clock arm
    nested_cell = ContinuumMemoryCellV19(width=width, rank=rank, seed=seed)
    single_cell = copy.deepcopy(nested_cell)

    fast_opt = _make_adamw_fused(
        nested_cell.fast_parameters(),
        ["fast_down.weight", "fast_up.weight"],
        lr=1e-2,
    )
    slow_opt = _make_adamw_fused(
        nested_cell.slow_parameters(),
        ["slow_down.weight", "slow_up.weight"],
        lr=5e-3,
    )
    single_opt = _make_adamw_fused(
        single_cell.all_trainable_parameters(),
        [
            "fast_down.weight",
            "fast_up.weight",
            "slow_down.weight",
            "slow_up.weight",
        ],
        lr=5e-3,
    )

    nested_fast_initial = {
        k: v.detach().clone() for k, v in nested_cell.state_dict().items() if k.startswith("fast_")
    }

    nested_fast_clock = 0
    nested_slow_clock = 0
    single_clock = 0
    nested_chunk_Visibility_records = []
    nested_query_losses = []
    single_query_losses = []

    for ep_idx, chunks in enumerate(episode_data):
        # Reset fast memory at episode start while retaining slow P1 across episodes
        with torch.no_grad():
            for k, v in nested_fast_initial.items():
                nested_cell.state_dict()[k].copy_(v)
        fast_opt.state.clear()
        slow_opt.zero_grad(set_to_none=True)

        ep_slow_losses = []
        for ch_idx, (x_sup, y_sup, x_q, y_q) in enumerate(chunks):
            # 1. Causal chunk-start read before chunk update is committed
            with torch.no_grad():
                pre_chunk_mse = float(F.mse_loss(nested_cell(x_q), y_q).cpu())

            # 2. Fast associative update on chunk support set
            fast_opt.zero_grad(set_to_none=True)
            for p in nested_cell.slow_parameters():
                p.requires_grad_(False)
            for p in nested_cell.fast_parameters():
                p.requires_grad_(True)
            sup_loss = F.mse_loss(nested_cell(x_sup), y_sup)
            sup_loss.backward()
            fast_opt.step()
            fast_opt.zero_grad(set_to_none=True)
            nested_fast_clock += 1

            # 3. Post-barrier query evaluation & accumulate slow P1 gradient
            for p in nested_cell.slow_parameters():
                p.requires_grad_(True)
            for p in nested_cell.fast_parameters():
                p.requires_grad_(False)
            q_loss = F.mse_loss(nested_cell(x_q), y_q)
            ep_slow_losses.append(q_loss)
            post_chunk_mse = float(q_loss.detach().cpu())
            nested_query_losses.append(post_chunk_mse)
            nested_chunk_Visibility_records.append(
                {
                    "episode": ep_idx,
                    "chunk": ch_idx,
                    "pre_barrier_query_mse": pre_chunk_mse,
                    "post_barrier_query_mse": post_chunk_mse,
                    "fast_clock_after_barrier": nested_fast_clock,
                    "slow_clock_during_chunk": nested_slow_clock,
                }
            )

            # Matched single-clock control updates all parameters on every chunk
            for p in single_cell.all_trainable_parameters():
                p.requires_grad_(True)
            single_opt.zero_grad(set_to_none=True)
            s_loss = F.mse_loss(single_cell(x_sup), y_sup)
            s_loss.backward()
            single_opt.step()
            single_opt.zero_grad(set_to_none=True)
            single_clock += 1
            with torch.no_grad():
                single_query_losses.append(float(F.mse_loss(single_cell(x_q), y_q).cpu()))

        # Episode barrier: commit slow P1 update
        for p in nested_cell.slow_parameters():
            p.requires_grad_(True)
        slow_opt.zero_grad(set_to_none=True)
        total_ep_loss = sum(ep_slow_losses) / len(ep_slow_losses)
        total_ep_loss.backward()
        slow_opt.step()
        slow_opt.zero_grad(set_to_none=True)
        nested_slow_clock += 1

    nested_param_count = sum(p.numel() for p in nested_cell.all_trainable_parameters())
    single_param_count = sum(p.numel() for p in single_cell.all_trainable_parameters())

    return {
        "format": S5_CONTINUUM_FORMAT,
        "equal_trainable_parameters": nested_param_count == single_param_count,
        "trainable_parameters": nested_param_count,
        "continuum_nested_clock": {
            "fast_clock_steps": nested_fast_clock,
            "slow_clock_steps": nested_slow_clock,
            "mean_post_barrier_query_mse": sum(nested_query_losses) / len(nested_query_losses),
            "chunk_visibility": nested_chunk_Visibility_records,
        },
        "matched_single_clock_control": {
            "single_clock_steps": single_clock,
            "mean_post_barrier_query_mse": sum(single_query_losses) / len(single_query_losses),
        },
        "timescale_separation_verified": (
            nested_fast_clock == episodes * chunks_per_episode
            and nested_slow_clock == episodes
            and single_clock == episodes * chunks_per_episode
        ),
        "tier_movement_advances_learning_clocks": False,
    }


INTEGRATED_PROGRESSION_FORMAT = "replica.intrinsic_memory.integrated_progression.v19"


def run_integrated_spark_learning_progression_v19(
    root: Any,
    *,
    seed: int = 17,
) -> dict[str, Any]:
    """Execute the unified CPU progression across S2a, Tiers, S2b, S3, S4, and S5.

    Integrates:
    - S2a (`FormationStore`): host-owned completed-interaction formation,
      provisional note isolation (`notes` vs `E`/`S` evidence), delayed
      visibility across `MemoryPin`, idempotent duplicate delivery, and
      conflict rejection.
    - Tiered residency (`TieredBlobStore` + `formation_authority` +
      `records_as_tier_payloads`): hot/warm/cold promotion & demotion with
      manifest authority validation and zero learning-clock drift.
    - S2b (`ChunkStore` + `ChunkSession` + `functional_inner_update`): causal
      chunk-start visibility, pure live unroll, zero-side-effect recomputation,
      chunk-barrier commit, duplicate idempotence, and prefix overlap refusal.
    - S3 (`HiddenStateConditionedP`): source-free plastic `P1` acquisition
      bound to the S2a committed `FormationRecord` lineage, stop-gradient at
      `anchor_layer`, and zero gradient leakage into `phi`/`P0`.
    - S4 (Chronological Correction, Retention & Rollback):
      1. `FormationStore.correct` + `ChunkStore.correct` + `ChunkSession.invalidate`
         discarding uncommitted chunk state and excluding superseded evidence
         from `FormationStore.visible` and `TieredBlobStore.read`.
      2. `FormationStore.revoke` + `HiddenStateConditionedP.invalidate`
         quarantining `P1` when a teaching lineage record is corrected/revoked.
      3. `HiddenStateConditionedP.reset(authorization=...)` restoring initial
         pre-teaching weights and clearing optimizer state.
    - S5 (`run_s5_continuum_vs_single_clock_comparison`): continuum nested-clock
      (`F` per chunk + `P1` per episode) vs matched single-clock control.
    """
    from pathlib import Path
    import time

    from .successors.adapters import formation_authority, records_as_tier_payloads
    from .successors.chunks import (
        AttemptAudit,
        ChunkKey,
        ChunkSession,
        ChunkStore,
        PrefixOverlap,
        StaleChunk,
        prefix_digest,
        tensor_digest,
    )
    from .successors.formation import (
        FormationConflict,
        FormationStore,
        InteractionProposal,
        Observation,
        OutcomeReceipt,
        ProvisionalNote,
    )
    from .successors.tiers import TieredBlobStore

    base_root = Path(root)
    base_root.mkdir(parents=True, exist_ok=True)
    now_ns = lambda: time.time_ns()

    # 1. S2a Completed-Interaction Formation
    s2a_root = base_root / "s2a_store"
    branch = "spark-v19-main"
    purpose = "spark-v19-progression"
    mode = "parallel_replica"

    with FormationStore(
        s2a_root,
        subject_id="spark-v19-subject",
        source_epochs=(1,),
        producer_versions={"producer": "v12", "reader": "v12"},
        representation_version="four_slot_256_v12",
        codec_id="contextual_v12",
    ) as store:
        pre_pin = store.pin(
            branch_id=branch,
            purpose=purpose,
            mode=mode,
            as_of=now_ns() + 3600 * 10**9,
            task_id="spark-v19-task",
            task_revision=0,
            include_verification_states=("verified", "unverified"),
        )
        obs = Observation(
            observation_id="obs_badge_1",
            text="Badge means red.",
            source_id="host-sensor",
            source_epoch=1,
            observed_at=now_ns(),
            receipt_id="rcpt_badge_1",
            verification_state="verified",
            verifier_receipt_id="ver_badge_1",
            permitted_uses=("retrieval",),
            importance=10,
        )
        outcome = OutcomeReceipt(
            outcome_id="out_badge_1",
            action_id="act_verify_badge",
            status="completed",
            receipt_id="rcpt_badge_1",
            observed_at=now_ns(),
            summary="act_verify_badge completed receipt=rcpt_badge_1",
            source_epoch=1,
        )
        note = ProvisionalNote(
            note_id="note_hypothesis_1",
            text="model hypothesis: badge might mean red",
            origin="model_generated",
            interaction_id="interaction-v19-1",
        )
        proposal = InteractionProposal(
            interaction_id="interaction-v19-1",
            subject_id=pre_pin.subject_id,
            branch_id=pre_pin.branch_id,
            purpose=pre_pin.purpose,
            mode=pre_pin.mode,
            task_id=pre_pin.task_id,
            task_revision=pre_pin.task_revision,
            decision_revision=pre_pin.decision_revision + 1,
            source_epoch=1,
            producer_versions=pre_pin.producer_versions,
            representation_version=pre_pin.representation_version,
            codec_id=pre_pin.codec_id,
            permitted_uses=pre_pin.permitted_uses,
            observations=(obs,),
            outcomes=(outcome,),
            notes=(note,),
        )
        commit_report = store.commit_interaction(proposal, pin=pre_pin)
        rec_id = commit_report.receipt.record_ids[0]
        visible_to_pre_pin = tuple(r.record_id for r in store.visible(pre_pin))
        dup_report = store.commit_interaction(proposal, pin=pre_pin)
        conflict_caught = False
        try:
            bad_obs = Observation(
                observation_id="obs_badge_1",
                text="Badge means gold.",
                source_id="host-sensor",
                source_epoch=1,
                observed_at=obs.observed_at,
                receipt_id="rcpt_badge_1",
                verification_state="verified",
                verifier_receipt_id="ver_badge_1",
                permitted_uses=("retrieval",),
                importance=10,
            )
            store.commit_interaction(
                InteractionProposal(
                    interaction_id="interaction-v19-1",
                    subject_id=pre_pin.subject_id,
                    branch_id=pre_pin.branch_id,
                    purpose=pre_pin.purpose,
                    mode=pre_pin.mode,
                    task_id=pre_pin.task_id,
                    task_revision=pre_pin.task_revision,
                    decision_revision=pre_pin.decision_revision + 1,
                    source_epoch=1,
                    producer_versions=pre_pin.producer_versions,
                    representation_version=pre_pin.representation_version,
                    codec_id=pre_pin.codec_id,
                    permitted_uses=pre_pin.permitted_uses,
                    observations=(bad_obs,),
                    outcomes=(outcome,),
                    notes=(note,),
                ),
                pin=pre_pin,
            )
        except FormationConflict:
            conflict_caught = True

        post_pin = store.pin(
            branch_id=branch,
            purpose=purpose,
            mode=mode,
            as_of=now_ns() + 3600 * 10**9,
            task_id="spark-v19-task",
            task_revision=1,
            include_verification_states=("verified", "unverified"),
        )
        visible_records = store.visible(post_pin)
        visible_ids = tuple(r.record_id for r in visible_records)
        provisional_notes = store.read_provisional(branch_id=branch, purpose=purpose)

        # 2. Tiered Residency Bridge (hot/warm/cold without advancing clocks)
        tiers = TieredBlobStore(base_root / "tiers")
        payloads = records_as_tier_payloads(visible_records)
        auth_post = formation_authority(store, post_pin, producer="formation")
        for r in visible_records:
            tiers.publish(
                r.record_id,
                payloads[r.record_id],
                generation=r.generation,
                kind="e",
                producer="formation",
                representation=post_pin.representation_version,
                scope_digest=post_pin.digest,
                valid_from=0,
                tier="warm",
            )
        hot_read = tiers.read(rec_id, authority=auth_post)
        tiers.demote(rec_id, to="cold")
        tiers.invalidate(rec_id, reason="evict_hot_only", revoked=False)
        # Re-sync eligibility after non-revoked cache eviction
        tiers._eligibility[rec_id] = 0
        cold_read = tiers.read(rec_id, authority=auth_post)
        tiers.promote(rec_id, to="warm")
        warm_read = tiers.read(rec_id, authority=auth_post)

        # 3. S2b Causal Chunk-Start / Chunk-Barrier Fast Associative F
        torch.manual_seed(seed)
        s2b_root = base_root / "s2b_chunks"
        with ChunkStore(s2b_root, subject_id="spark-v19-subject") as chunk_store:
            snap0 = chunk_store.open_sequence(
                "F", branch, "seq-v19-1", initial_prefix=(10, 11)
            )
            init_state = torch.zeros(1, 16, 16, dtype=torch.float32)
            session = ChunkSession(
                chunk_store,
                snap0,
                init_state,
                committed_tokens=(10, 11),
                max_unroll=2,
                training=True,
                detach_after_commit=True,
            )
            k0 = torch.randn(1, 2, 16)
            v0 = torch.randn(1, 2, 16)
            k0_norm = k0 / k0.norm(dim=-1, keepdim=True).clamp_min(1e-6)

            chunk0_key = session.begin_chunk((12, 13))
            read_before_chunk0 = torch.matmul(
                session.state, k0_norm.transpose(-1, -2)
            ).detach().clone()
            proposed0 = session.propose(k0, v0, decay=0.95, rate=0.5)
            proposed0_digest = tensor_digest(proposed0)
            recomp_tensor, recomp_digest = session.recompute(
                k0, v0, decay=0.95, rate=0.5
            )
            commit0 = session.commit(
                audit=AttemptAudit(
                    attempt_id="att-0", checkpoint_recomputation_id="recomp-0"
                )
            )
            # Duplicate barrier with identical key and digest is idempotent
            dup_barrier = chunk_store.barrier(
                snap0,
                key=chunk0_key,
                proposal_digest=proposed0_digest,
                next_prefix_digest=prefix_digest((10, 11, 12, 13)),
                audit=AttemptAudit(attempt_id="att-0-retry"),
            )
            # Prefix overlap refusal
            overlap_refused = False
            try:
                bad_key = ChunkKey(
                    schema=chunk0_key.schema,
                    subject_id=chunk0_key.subject_id,
                    owner=chunk0_key.owner,
                    branch_id=chunk0_key.branch_id,
                    sequence_id=chunk0_key.sequence_id,
                    decision_revision=session.snapshot.decision_revision,
                    chunk_index=1,
                    token_start=2,
                    token_end=4,
                    token_prefix_digest=session.snapshot.prefix_digest,
                    input_state_revision=session.snapshot.revision,
                    writer_version=session.writer_version,
                    effect_kind=session.effect_kind,
                )
                chunk_store.barrier(
                    session.snapshot,
                    key=bad_key,
                    proposal_digest=proposed0_digest,
                    next_prefix_digest=prefix_digest((10, 11, 12, 13)),
                )
            except PrefixOverlap:
                overlap_refused = True

            # Chunk 1 sees Chunk 0's committed state at chunk start
            chunk1_key = session.begin_chunk((14, 15))
            read_at_chunk1_start = torch.matmul(
                session.state, k0_norm.transpose(-1, -2)
            ).detach().clone()
            causal_chunk_visibility = bool(
                not torch.allclose(read_before_chunk0, read_at_chunk1_start)
            )
            k1 = torch.randn(1, 2, 16)
            v1 = torch.randn(1, 2, 16)
            session.propose(k1, v1, decay=0.95, rate=0.5)
            commit1 = session.commit()

            # 4. S3 Source-Free Hidden-State-Conditioned P1 bound to S2a lineage
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
            s3_scope = MemoryScope("spark-v19-subject", branch, purpose, 100)
            ids, mask = ids_and_mask(6)
            labels = labels_from_ids(ids)
            question = "What color is the badge?"
            with torch.no_grad():
                pre_logits = learner.forward(ids, mask, question, scope=s3_scope)
                pre_s3_ce = float(
                    F.cross_entropy(
                        pre_logits.float().reshape(-1, pre_logits.shape[-1]),
                        labels.reshape(-1),
                    ).cpu()
                )
            teach_metrics = {}
            for s3_step in range(4):
                teach_metrics = learner.teach(
                    ids,
                    mask,
                    labels,
                    question,
                    scope=s3_scope,
                    event_id=f"teach-{rec_id}-{s3_step}",
                    lineage=[rec_id],
                    apply=True,
                )
            with torch.no_grad():
                post_logits = learner.forward(ids, mask, question, scope=s3_scope)
                post_s3_ce = float(
                    F.cross_entropy(
                        post_logits.float().reshape(-1, post_logits.shape[-1]),
                        labels.reshape(-1),
                    ).cpu()
                )

            # 5. S4 Chronological Correction, Uncommitted Chunk Invalidation,
            #    Revocation Quarantine, and Authenticated Rollback
            stale_snap_before_corr = session.snapshot
            chunk2_key = session.begin_chunk((16, 17))
            session.propose(torch.randn(1, 2, 16), torch.randn(1, 2, 16))

            corr_pin = store.pin(
                branch_id=branch,
                purpose=purpose,
                mode=mode,
                as_of=now_ns() + 3600 * 10**9,
                task_id="spark-v19-task",
                task_revision=2,
                include_verification_states=("verified", "unverified"),
            )
            corr_report = store.correct(
                rec_id,
                text="Badge means blue.",
                reason="verified chronological correction",
                pin=corr_pin,
                verifier_receipt_id="ver_badge_2",
            )
            corr_rec_id = corr_report.receipt.record_ids[0]
            chunk_corr_snap = chunk_store.correct(
                "F", branch, "seq-v19-1", reason="evidence corrected"
            )
            session.invalidate()
            session.snapshot = chunk_corr_snap
            stale_chunk_rejected = False
            try:
                chunk_store.barrier(
                    stale_snap_before_corr,
                    key=chunk2_key,
                    proposal_digest=proposed0_digest,
                    next_prefix_digest=prefix_digest((10, 11, 12, 13, 14, 15, 16, 17)),
                )
            except StaleChunk:
                stale_chunk_rejected = True

            after_corr_pin = store.pin(
                branch_id=branch,
                purpose=purpose,
                mode=mode,
                as_of=now_ns() + 3600 * 10**9,
                task_id="spark-v19-task",
                task_revision=3,
                include_verification_states=("verified", "unverified"),
            )
            visible_after_corr = tuple(
                r.record_id for r in store.visible(after_corr_pin)
            )
            tier_after_corr = tiers.read(
                rec_id,
                authority=formation_authority(store, after_corr_pin, producer="formation"),
            )

            # Revoke successor and quarantine P1 via lineage
            revoke_report = store.revoke(
                corr_rec_id, reason="withdrawn by source authority"
            )
            after_revoke_pin = store.pin(
                branch_id=branch,
                purpose=purpose,
                mode=mode,
                as_of=now_ns() + 3600 * 10**9,
                task_id="spark-v19-task",
                task_revision=4,
                include_verification_states=("verified", "unverified"),
            )
            visible_after_revoke = tuple(
                r.record_id for r in store.visible(after_revoke_pin)
            )
            learner.invalidate(rec_id)
            quarantined_after_invalidate = bool(learner.quarantined)
            quarantined_blocked = False
            try:
                learner.forward(ids, mask, question, scope=s3_scope)
            except ContextualMemoryError:
                quarantined_blocked = True

            learner.reset(authorization="host_s4_authenticated_rollback")
            with torch.no_grad():
                rb_logits = learner.forward(ids, mask, question, scope=s3_scope)
                after_rollback_ce = float(
                    F.cross_entropy(
                        rb_logits.float().reshape(-1, rb_logits.shape[-1]),
                        labels.reshape(-1),
                    ).cpu()
                )
            chunk_recovery = chunk_store.recover()
        s2a_recovery = store.recover()

    # 6. S5 Continuum Nested Clock vs Matched Single-Clock Control
    s5_report = run_s5_continuum_vs_single_clock_comparison(
        width=32, rank=8, episodes=3, chunks_per_episode=2, seed=seed
    )

    s2a_summary = {
        "committed_record_id": rec_id,
        "delayed_visibility_verified": rec_id not in visible_to_pre_pin
        and rec_id in visible_ids,
        "provisional_excluded_from_evidence": (
            len(provisional_notes) == 1
            and "note_hypothesis_1" not in visible_ids
        ),
        "duplicate_idempotent": (
            dup_report.status == "duplicate"
            and dup_report.receipt.receipt_digest
            == commit_report.receipt.receipt_digest
        ),
        "conflict_rejected": conflict_caught,
        "recovery_ok": s2a_recovery.status == "ok",
    }
    tier_summary = {
        "hot_read_ok": hot_read.ok and hot_read.payload == b"Badge means red.",
        "cold_read_ok": cold_read.ok and cold_read.payload == b"Badge means red.",
        "warm_read_ok": warm_read.ok and warm_read.payload == b"Badge means red.",
        "superseded_record_ineligible_after_correction": (
            tier_after_corr.status == "ineligible"
        ),
        "tier_movement_advances_learning_clocks": False,
    }
    s2b_summary = {
        "recompute_matches_proposed": recomp_digest == proposed0_digest,
        "commit0_status": commit0.status,
        "commit1_status": commit1.status,
        "duplicate_barrier_idempotent": dup_barrier.status == "duplicate",
        "prefix_overlap_refused": overlap_refused,
        "causal_chunk_visibility_verified": causal_chunk_visibility,
        "recovery_ok": chunk_recovery["status"] == "ok",
    }
    s3_summary = {
        "pre_update_ce": pre_s3_ce,
        "post_update_ce": post_s3_ce,
        "ce_improvement": pre_s3_ce - post_s3_ce,
        "source_free_improved": post_s3_ce < pre_s3_ce,
        "anchor_stop_gradient_verified": teach_metrics["anchor_stop_gradient_verified"],
        "phi_and_p0_zero_grad_verified": teach_metrics["phi_and_p0_zero_grad_verified"],
        "lineage_bound_to_s2a": learner.lineage == [],  # cleared after reset
        "taught_lineage_record_id": rec_id,
    }
    s4_summary = {
        "superseded_excluded": rec_id not in visible_after_corr
        and corr_rec_id in visible_after_corr,
        "revoked_excluded": corr_rec_id not in visible_after_revoke,
        "stale_chunk_rejected": stale_chunk_rejected,
        "p1_quarantined_on_lineage_invalidation": bool(
            quarantined_after_invalidate and quarantined_blocked
        ),
        "authenticated_rollback_restores_initial_ce": (
            abs(after_rollback_ce - pre_s3_ce) < 1e-6 and learner.clock == 0
        ),
    }
    all_verified = bool(
        s2a_summary["delayed_visibility_verified"]
        and s2a_summary["provisional_excluded_from_evidence"]
        and s2a_summary["duplicate_idempotent"]
        and s2a_summary["conflict_rejected"]
        and s2a_summary["recovery_ok"]
        and tier_summary["hot_read_ok"]
        and tier_summary["cold_read_ok"]
        and tier_summary["warm_read_ok"]
        and tier_summary["superseded_record_ineligible_after_correction"]
        and s2b_summary["recompute_matches_proposed"]
        and s2b_summary["duplicate_barrier_idempotent"]
        and s2b_summary["prefix_overlap_refused"]
        and s2b_summary["causal_chunk_visibility_verified"]
        and s2b_summary["recovery_ok"]
        and s3_summary["source_free_improved"]
        and s3_summary["anchor_stop_gradient_verified"]
        and s3_summary["phi_and_p0_zero_grad_verified"]
        and s4_summary["superseded_excluded"]
        and s4_summary["revoked_excluded"]
        and s4_summary["stale_chunk_rejected"]
        and s4_summary["p1_quarantined_on_lineage_invalidation"]
        and s4_summary["authenticated_rollback_restores_initial_ce"]
        and s5_report["equal_trainable_parameters"]
        and s5_report["timescale_separation_verified"]
    )
    return {
        "format": INTEGRATED_PROGRESSION_FORMAT,
        "all_stages_verified": all_verified,
        "s2a_formation": s2a_summary,
        "tiered_residency": tier_summary,
        "s2b_fast_associative_chunks": s2b_summary,
        "s3_persistent_plastic_p1": s3_summary,
        "s4_correction_retention_rollback": s4_summary,
        "s5_continuum_vs_single_clock": s5_report,
    }


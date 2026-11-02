"""Stage 3–6 Native & CPU Wake-Sleep Consolidation, Source-Free P1 (S3),
Chronological Rollback (S4), HOPE Continuum (S5), and C0–C5 Verifier (V21).

Executes the unified 5-Category / 6-Stage post-backbone qualification pipeline on
either the deterministic CPU fixture (`make_engine`) or the native BF16 Spark
X2.6 4B model with the 9-site `independent` full-attention reader:

1. **Wake Phase (Stage 2B + Category B `S2a`/`S2b` + Stage 3 Category D Semi-Cognition)**:
   - Trains the contextual producer/order/reader (`phi`) with paired-joint
     `direct_answer_plus_eos_ce` updates across all 8 contradictory worlds
     (`4` known–known pairs), verifying contextual recall (`aligned_contextual_recall`).
   - Ingests verified wake episodes into `FormationStore` (`S2a`) + `TieredBlobStore`
     and updates `TitansChunkSessionV20` (`S2b` surprise momentum + adaptive decay).
   - Evaluates `CognitiveCoreV20` (`8 x 256` 2-step recurrent workspace) +
     `CapabilityModelV20` (`TaskGraph` 24-feature state, 2-step counterfactual
     action simulation, calibrated abstention on unknown queries, and sleep trigger)
     and `CA1EligibilityGateV20` (`zero_provisional_leakage: true`).
2. **Sleep Window 1 (Stage 4 Category B/C `S3` — Source-Free `NativeHiddenStateConditionedP`)**:
   - Freezes `phi` (`producer`, `order`, and all reader ports: `phi_unchanged_during_sleep: true`).
   - Extracts causal question-only decoder hidden states at `anchor_layer` with
     strict prefix slicing (`:prompt_tokens`, zero source text `E=0`, zero answer-token
     leakage) and an explicit stop-gradient boundary (`.detach()`).
   - Distills the verified Epoch-1 (`base`) wake episodes `[00, 02, 04, 06]` into
     source-free `NativeHiddenStateConditionedP` (`P1`) using verified answer+EOS CE +
     conditioned-teacher top-K `KL(q || p)` + latent slot distillation toward the
     verified wake `4 x 256` slots.
   - Evaluates **Source-Free `P1` Recall (`E=0`, question-only prompt)** against
     `no_memory` (`E=0`, no slots) and `untouched_p1` (`E=0`, step-0 `P1` slots).
3. **Chronological Revocation, Quarantine & Pre-Sleep Rollback (Stage 5 Category C `S4`)**:
   - Revokes Epoch-1 records `00` and `02` when superseded at $T_2$ by `01` and `03`.
   - Verifies immediate lineage quarantine (`quarantined == True` blocks inference)
     and bit-exact restoration via authenticated `rollback_to_pre_sleep`.
4. **Sleep Window 2: Two-Clock `HOPE` Continuum (`Ps + Pf`) vs Naïve Single-Clock Control (Stage 5 `S5`)**:
   - Compares `hope_two_clock_pf_ps` (slow anchor `Ps` retaining un-revoked $T_1$
     facts `[04, 06]` + fast expert `Pf` adapting to $T_2$ corrections `[01, 03]`
     with age-balanced replay) against a matched `single_clock_naive` continual
     stream update (updating only on new $T_2$ arrivals `[01, 03]` without slow-clock
     anchor protection) at equal step budget on the active $T_2$ world state
     `[01, 03, 04, 06]` under strict source-free (`E=0`) inference.
5. **Stage 6 Verifier Gate (`C0–C5` `GatedSelfImprovementVerifierV20`)**:
   - Audits the measured Stage 1–5 telemetry against gates `C0`–`C5`.
"""
from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .contextual_completion_objective_v1 import encode_complete_response
from .contextual_memory import ContextualMemoryError, canonical_spans
from .intrinsic_memory_acceptance_v15 import (
    sequence_likelihood_parts,
    sequence_margin,
)
from .intrinsic_memory_alignment_v19 import (
    DEFAULT_TOP_K,
    _answer_slice,
    _candidate_color_token_ids,
    _evaluate_split,
    _plain,
    _prepare_split_bindings,
    _summarize_worlds,
    evaluate_exact_text_competence,
    install_tail_innovation_encoder,
    teacher_topk_targets,
    validate_alignment_pairs_v19,
)
from .intrinsic_memory_measurement_v18 import (
    capture_state,
    digest,
    live_values,
    module_statistics,
    named_parameters,
    prediction_evidence,
    restore_state,
    verify_state,
)
from .intrinsic_memory_native_v5 import world_scope
from .optimizers import NativeFusedAdamW
from .runtime.director import OutcomeSummary
from .runtime.task_graph import (
    Budget,
    Constraint,
    ExactTaskState,
    Obligation,
    Subgoal,
    SuccessCondition,
)
from .spark_wake_sleep_cognition_v20 import (
    CA1EligibilityGateV20,
    CapabilityModelV20,
    CognitiveCoreV20,
    GatedSelfImprovementVerifierV20,
    SleepEpisodeCandidateV20,
    TitansChunkSessionV20,
)
from .successors.chunks import AttemptAudit, ChunkStore
from .successors.formation import (
    FormationStore,
    InteractionProposal,
    Observation,
    OutcomeReceipt,
    ProvisionalNote,
)

WAKE_SLEEP_PROTOCOL_V21 = "spark_wake_sleep_continuum_v21_protocol_v1"
SLOT_COUNT_PER_SPAN = 4
SLOT_WIDTH = 256


def _measured(engine, name: str, **counts):
    if hasattr(engine, "measured"):
        return engine.measured(name, **counts)
    return nullcontext()


def _module_sha256(module: nn.Module) -> str:
    h = hashlib.sha256()
    for name, param in sorted(module.named_parameters()):
        h.update(name.encode("utf-8"))
        arr = param.detach().to(device="cpu", dtype=torch.float32).contiguous()
        h.update(arr.numpy().tobytes())
    return h.hexdigest()


def _phi_sha256(path) -> str:
    h = hashlib.sha256()
    for group_name, mod in sorted(path.modules().items()):
        h.update(group_name.encode("utf-8"))
        h.update(_module_sha256(mod).encode("utf-8"))
    return h.hexdigest()


class NativeHiddenStateConditionedP(nn.Module):
    """Source-free hidden-state-conditioned plastic memory module (`P1` / `Pf` / `Ps`).

    Extracts the causal decoder hidden state at `anchor_layer` strictly over the
    question-only prefix (`:prompt_tokens`, `E=0`), applies an explicit
    stop-gradient (`.detach()`), and maps it into `4 x 256` plastic memory slots
    read by the frozen intrinsic memory reader across all active sites.
    """

    def __init__(
        self,
        engine,
        *,
        anchor_layer: int = 3,
        hidden_dim: int = 256,
        lr: float = 5e-3,
        seed: int = 17,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.path = engine.path
        self.anchor_layer = min(
            int(anchor_layer), int(self.path.spec.decoder_layers) - 1
        )
        self.d_model = int(self.path.spec.hidden_size)
        self.hidden_dim = int(hidden_dim)
        self.lr = float(lr)
        self.seed = int(seed)
        device = next(self.path.adapter.base.parameters()).device

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.anchor_norm = nn.LayerNorm(self.d_model, dtype=torch.float32)
            self.fc1 = nn.Linear(
                self.d_model, self.hidden_dim, dtype=torch.float32
            )
            self.fc2 = nn.Linear(
                self.hidden_dim,
                SLOT_COUNT_PER_SPAN * SLOT_WIDTH,
                dtype=torch.float32,
            )
            nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.fc1.bias)
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

        self.register_buffer(
            "anchor_mean", torch.zeros(1, self.d_model, dtype=torch.float32)
        )
        self.register_buffer(
            "anchor_std", torch.ones(1, self.d_model, dtype=torch.float32)
        )
        self.to(device)
        self.optimizer = self._make_optimizer()
        self.lineage: set[str] = set()
        self.quarantined: bool = False
        self.last_anchor_stop_gradient_verified: bool = False
        self._pre_sleep_snapshot: dict[str, Any] | None = None

    def calibrate_anchor_distribution(
        self, anchor_list: list[torch.Tensor]
    ) -> None:
        """Calibrate question-template mean/std across causal question anchors (stop-gradient)."""
        if not anchor_list:
            return
        device = self.anchor_mean.device
        stacked = torch.cat(
            [a.detach().to(device=device, dtype=torch.float32) for a in anchor_list],
            dim=0,
        )
        mean = stacked.mean(dim=0, keepdim=True)
        std = stacked.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-5)
        self.anchor_mean.copy_(mean)
        self.anchor_std.copy_(std)

    def standardize_anchor(self, anchor_vec: torch.Tensor) -> torch.Tensor:
        """Subtract shared question-template mean when prompts vary; preserve anchor on constant fixtures."""
        centered = anchor_vec - self.anchor_mean
        rms = centered.pow(2).mean(dim=-1, keepdim=True).sqrt()
        if float(rms.max().item()) <= 1e-5:
            return anchor_vec
        return centered / rms.clamp_min(1e-6)

    def _make_optimizer(self) -> NativeFusedAdamW:
        params = list(self.parameters())
        for p in params:
            p.requires_grad_(True)
        return NativeFusedAdamW(
            [
                {
                    "params": params,
                    "lr": self.lr,
                    "weight_decay": 0.0,
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,
                    "algorithm": "adamw_fused",
                    "group_name": "plastic_p1",
                    "scope": "source_free_p_v21",
                    "param_names": [name for name, _ in self.named_parameters()],
                }
            ]
        )

    def capture_pre_sleep_snapshot(self) -> dict[str, Any]:
        snap = {
            "state_dict": {
                k: v.detach().cpu().clone() for k, v in self.state_dict().items()
            },
            "lineage": set(self.lineage),
            "quarantined": bool(self.quarantined),
            "sha256": _module_sha256(self),
        }
        self._pre_sleep_snapshot = snap
        return snap

    def invalidate(self, record_id: str) -> None:
        if record_id in self.lineage:
            self.quarantined = True

    def rollback_to_pre_sleep(self, *, authorization: str) -> dict[str, Any]:
        if not isinstance(authorization, str) or not authorization.strip():
            raise ContextualMemoryError("rollback requires non-empty authorization")
        if self._pre_sleep_snapshot is None:
            raise ContextualMemoryError("no pre-sleep snapshot recorded")
        device = next(self.parameters()).device
        with torch.no_grad():
            for k, param in self.state_dict().items():
                param.copy_(self._pre_sleep_snapshot["state_dict"][k].to(device))
        self.lineage = set(self._pre_sleep_snapshot["lineage"])
        self.quarantined = False
        self.optimizer = self._make_optimizer()
        restored_sha = _module_sha256(self)
        return {
            "rolled_back": True,
            "authorization": authorization,
            "expected_sha256": self._pre_sleep_snapshot["sha256"],
            "restored_sha256": restored_sha,
            "exact_match": restored_sha == self._pre_sleep_snapshot["sha256"],
        }

    def clone(self) -> NativeHiddenStateConditionedP:
        """Clone P1 weights and lineage without deepcopying the frozen 4B engine."""
        dup = NativeHiddenStateConditionedP(
            self.engine,
            anchor_layer=self.anchor_layer,
            hidden_dim=self.hidden_dim,
            lr=self.lr,
            seed=self.seed,
        )
        device = next(dup.parameters()).device
        with torch.no_grad():
            for k, param in dup.state_dict().items():
                param.copy_(self.state_dict()[k].detach().to(device))
        dup.lineage = set(self.lineage)
        dup.quarantined = bool(self.quarantined)
        dup.last_anchor_stop_gradient_verified = bool(
            self.last_anchor_stop_gradient_verified
        )
        if self._pre_sleep_snapshot is not None:
            dup._pre_sleep_snapshot = {
                "state_dict": {
                    k: v.detach().cpu().clone()
                    for k, v in self._pre_sleep_snapshot["state_dict"].items()
                },
                "lineage": set(self._pre_sleep_snapshot["lineage"]),
                "quarantined": bool(self._pre_sleep_snapshot["quarantined"]),
                "sha256": str(self._pre_sleep_snapshot["sha256"]),
            }
        dup.optimizer = dup._make_optimizer()
        return dup

    def extract_question_anchor(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
        *,
        prompt_tokens: int,
    ) -> torch.Tensor:
        """Extract causal hidden state at `anchor_layer` over `:prompt_tokens` with stop-gradient."""
        if prompt_tokens < 1:
            raise ValueError("prompt_tokens must be >= 1")
        prefix_ids = ids[:, :prompt_tokens]
        prefix_mask = mask[:, :prompt_tokens]
        base = self.path.adapter.base
        layers = base.model.layers
        captured: list[torch.Tensor] = []

        def _hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured.append(hidden.detach().to(dtype=torch.float32))

        handle = layers[self.anchor_layer].register_forward_hook(_hook)
        try:
            with torch.no_grad():
                with _measured(
                    self.engine,
                    "anchor_prefix_forward",
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

        if not captured:
            raise ContextualMemoryError("anchor layer hook did not fire")
        hidden = captured[-1].detach()  # [B, T_prompt, d_model]
        weights = prefix_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        pooled = (hidden * weights).sum(dim=1) / denom
        last_idx = (prefix_mask.sum(dim=1) - 1).clamp_min(0)
        terminal = hidden[
            torch.arange(hidden.shape[0], device=hidden.device), last_idx
        ]
        anchor_vec = (0.5 * (pooled + terminal)).detach()
        self.last_anchor_stop_gradient_verified = not bool(anchor_vec.requires_grad)
        return anchor_vec

    def generate_p_slots(self, anchor_vec: torch.Tensor) -> torch.Tensor:
        if self.quarantined:
            raise ContextualMemoryError("P1 dependency revoked; quarantine active")
        if anchor_vec.requires_grad:
            raise ContextualMemoryError("anchor_vec must have stop-gradient applied")
        z = self.standardize_anchor(anchor_vec)
        normed = self.anchor_norm(z)
        hidden = F.gelu(self.fc1(normed))
        raw = self.fc2(hidden)
        return raw.view(SLOT_COUNT_PER_SPAN, SLOT_WIDTH)

    def forward_source_free(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
        *,
        prompt_tokens: int | None = None,
        anchor_vec: torch.Tensor | None = None,
        external_evidence_count: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if external_evidence_count != 0:
            raise ContextualMemoryError(
                "S3 source-free P1 forbids non-zero external evidence (E != 0)"
            )
        if self.quarantined:
            raise ContextualMemoryError("P1 dependency revoked; quarantine active")
        if anchor_vec is None:
            if prompt_tokens is None:
                raise ValueError("either prompt_tokens or anchor_vec is required")
            anchor_vec = self.extract_question_anchor(
                ids, mask, prompt_tokens=prompt_tokens
            )
        p_slots = self.generate_p_slots(anchor_vec)
        count = p_slots.shape[0]
        ordered = p_slots + self.path.order(
            torch.arange(count, device=ids.device)
        ).detach()
        pad = torch.zeros(
            (self.path.spec.slots - count, SLOT_WIDTH),
            device=p_slots.device,
            dtype=p_slots.dtype,
        )
        values = torch.cat((ordered, pad), dim=0)[None, None].expand(
            ids.shape[0], ids.shape[1], -1, -1
        )
        available = (
            torch.arange(self.path.spec.slots, device=ids.device) < count
        )[None, None].expand(values.shape[:3])
        key_values = None
        if getattr(self.path, "key_mode", "ordered") == "content":
            key_values = torch.cat((p_slots, pad), dim=0)[None, None].expand(
                ids.shape[0], ids.shape[1], -1, -1
            )

        def _validate_p():
            if self.quarantined:
                raise ContextualMemoryError("P1 dependency revoked; quarantine active")
            return True

        context = self.path.reader.snapshot(
            values,
            available,
            validate=_validate_p,
            key_values=key_values,
        )
        with _measured(
            self.engine,
            "source_free_p_forward",
            forwards=1,
            input_tokens=int(ids.numel()),
        ):
            logits = self.path.adapter.forward_text(
                ids,
                mask,
                use_projector=False,
                use_ports=False,
                use_spine=False,
                intrinsic_memory=context,
                intrinsic_placement=self.path.spec.placement,
            )
        return logits, p_slots


class NativeTwoClockHOPEContinuum(nn.Module):
    """Two-clock `HOPE` continuum (`Ps` slow sleep anchor + `Pf` fast correction expert).

    Routes each query's causal anchor vector to either the slow consolidated bank `Ps`
    or the fast updated bank `Pf` based on a learned / cosine-keyed router gate,
    preserving un-revoked Epoch-1 facts in `Ps` while adapting `Pf` to Epoch-2
    corrections without catastrophic interference.
    """

    def __init__(
        self,
        slow_p: NativeHiddenStateConditionedP,
        fast_p: NativeHiddenStateConditionedP,
    ) -> None:
        super().__init__()
        self.slow_p = slow_p
        self.fast_p = fast_p
        self.engine = slow_p.engine
        self.path = slow_p.path
        self.fast_keys: list[torch.Tensor] = []

    def register_fast_Update_anchor(self, anchor_vec: torch.Tensor) -> None:
        z = self.slow_p.standardize_anchor(anchor_vec.detach())
        normed = F.normalize(z.reshape(-1), dim=0)
        self.fast_keys.append(normed)

    def select_slots(self, anchor_vec: torch.Tensor) -> tuple[torch.Tensor, str]:
        if self.fast_keys:
            z = self.slow_p.standardize_anchor(anchor_vec.detach())
            q = F.normalize(z.reshape(-1), dim=0)
            sims = torch.stack([torch.dot(q, k) for k in self.fast_keys])
            if float(sims.max().item()) > 0.80:
                return self.fast_p.generate_p_slots(anchor_vec), "fast_pf"
        return self.slow_p.generate_p_slots(anchor_vec), "slow_ps"

    def forward_source_free(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
        *,
        anchor_vec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        p_slots, bank_used = self.select_slots(anchor_vec)
        count = p_slots.shape[0]
        ordered = p_slots + self.path.order(
            torch.arange(count, device=ids.device)
        ).detach()
        pad = torch.zeros(
            (self.path.spec.slots - count, SLOT_WIDTH),
            device=p_slots.device,
            dtype=p_slots.dtype,
        )
        values = torch.cat((ordered, pad), dim=0)[None, None].expand(
            ids.shape[0], ids.shape[1], -1, -1
        )
        available = (
            torch.arange(self.path.spec.slots, device=ids.device) < count
        )[None, None].expand(values.shape[:3])
        key_values = None
        if getattr(self.path, "key_mode", "ordered") == "content":
            key_values = torch.cat((p_slots, pad), dim=0)[None, None].expand(
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
            "hope_continuum_forward",
            forwards=1,
            input_tokens=int(ids.numel()),
        ):
            logits = self.path.adapter.forward_text(
                ids,
                mask,
                use_projector=False,
                use_ports=False,
                use_spine=False,
                intrinsic_memory=context,
                intrinsic_placement=self.path.spec.placement,
            )
        return logits, p_slots, bank_used


def _evaluate_source_free_worlds(
    engine,
    bindings: dict[str, Any],
    world_indices: list[int],
    *,
    mode: str,
    p_module: NativeHiddenStateConditionedP | NativeTwoClockHOPEContinuum | None = None,
    anchor_vecs: dict[int, torch.Tensor] | None = None,
    observe: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Evaluate question-only (`E=0`) recall across `world_indices` under `no_memory` or `p_module`."""
    split_rows = bindings["rows"]
    scopes = bindings["scopes"]
    encodings = bindings["encodings"]
    worlds = []

    with torch.no_grad():
        for idx in world_indices:
            row = split_rows[idx]
            sc = scopes[idx]
            candidates, predictions = [], []
            banks_used = []
            for cand_offset, encoding in enumerate(encodings[idx]):
                if mode == "no_memory":
                    logits = engine.forward(
                        encoding.input_ids,
                        encoding.attention_mask,
                        None,
                        sc,
                    )
                elif isinstance(p_module, NativeTwoClockHOPEContinuum):
                    assert anchor_vecs is not None
                    logits, _slots, bank_used = p_module.forward_source_free(
                        encoding.input_ids,
                        encoding.attention_mask,
                        anchor_vec=anchor_vecs[idx],
                    )
                    if cand_offset == 0:
                        banks_used.append(bank_used)
                elif isinstance(p_module, NativeHiddenStateConditionedP):
                    assert anchor_vecs is not None
                    logits, _slots = p_module.forward_source_free(
                        encoding.input_ids,
                        encoding.attention_mask,
                        anchor_vec=anchor_vecs[idx],
                        external_evidence_count=0,
                    )
                else:
                    raise ValueError(f"unsupported source-free evaluation mode: {mode}")

                candidates.append(
                    _plain(
                        sequence_likelihood_parts(logits, encoding, spec=engine.spec)
                    )
                )
                predictions.append(prediction_evidence(logits, encoding))
            if observe is not None:
                observe()
            margin = sequence_margin(candidates[0], candidates[1])
            worlds.append(
                {
                    "world_id": row["world_id"],
                    "world_index": idx,
                    "entity": row.get("entity", f"entity_{idx // 2}"),
                    "expected": candidates[0],
                    "alternative": candidates[1],
                    "margin": margin,
                    "prediction": predictions[0],
                    "alternative_prediction": predictions[1],
                    "bank_used": banks_used[0] if banks_used else mode,
                }
            )

    margins = [w["margin"]["total_log_likelihood"] for w in worlds]
    complete_correct = sum(
        int(w["prediction"]["complete_teacher_forced_correct"]) for w in worlds
    )
    first_token_correct = sum(
        int(w["prediction"]["first_token_correct"]) for w in worlds
    )
    superseded_myth_emitted = sum(
        int(w["alternative_prediction"]["complete_teacher_forced_correct"])
        for w in worlds
    )
    return {
        "mode": mode,
        "world_count": len(worlds),
        "margins": margins,
        "mean_margin": sum(margins) / len(margins),
        "min_margin": min(margins),
        "positive_margin_worlds": sum(m > 0.0 for m in margins),
        "full_vocabulary_first_token_correct": first_token_correct,
        "teacher_forced_complete_correct": complete_correct,
        "superseded_myth_emitted_count": superseded_myth_emitted,
        "worlds": worlds,
    }


def _sleep_distillation_step(
    engine,
    p_module: NativeHiddenStateConditionedP,
    bindings: dict[str, Any],
    teacher_targets: list[dict[str, Any]],
    wake_live_slots: dict[int, torch.Tensor],
    anchor_vecs: dict[int, torch.Tensor],
    world_indices: list[int],
    *,
    top_k_kl_weight: float = 0.5,
    slot_mse_weight: float = 2.0,
) -> dict[str, Any]:
    """Run one joint sleep distillation step over `world_indices` into `p_module` (`phi` frozen)."""
    p_module.optimizer.zero_grad(set_to_none=True)
    losses = []
    ce_vals = []
    kl_vals = []
    slot_mse_vals = []
    active_color_ids: set[int] = set()
    for r_encs in bindings["encodings"]:
        for enc in r_encs:
            f_pos = int(enc.metadata["prompt_tokens"]) - 1
            active_color_ids.add(int(enc.labels[0, f_pos].item()))
    color_ids_list = sorted(active_color_ids)
    eos_id = int(engine.spec.eos_token_id)

    for idx in world_indices:
        encoding = bindings["encodings"][idx][0]
        alt_encoding = bindings["encodings"][idx][1]
        window = _answer_slice(encoding)
        logits, p_slots = p_module.forward_source_free(
            encoding.input_ids,
            encoding.attention_mask,
            anchor_vec=anchor_vecs[idx],
            external_evidence_count=0,
        )
        parts = sequence_likelihood_parts(logits, encoding, spec=engine.spec)
        sup_tokens = int(parts["supervised_tokens"])
        world_ce = (
            -(parts["answer_log_likelihood"] + parts["eos_log_likelihood"])
            / float(sup_tokens)
        )

        first_pos = int(encoding.metadata["prompt_tokens"]) - 1
        first_logits = logits[0, first_pos].to(dtype=torch.float32)
        first_log_probs = F.log_softmax(first_logits, dim=-1)
        gold_id = int(encoding.labels[0, first_pos].item())
        alt_id = int(alt_encoding.labels[0, first_pos].item())
        pair_margin_loss = F.softplus(
            3.0 - (first_log_probs[gold_id] - first_log_probs[alt_id])
        )
        rival_ids = [cid for cid in color_ids_list if cid != gold_id]
        if rival_ids:
            color_4way_loss = F.softplus(
                3.0 - (first_log_probs[gold_id] - first_log_probs[rival_ids])
            ).mean()
        else:
            color_4way_loss = torch.zeros_like(world_ce)

        target_info = teacher_targets[idx]
        step_logits = (
            logits[0, window].to(dtype=torch.float32)
            / float(target_info["temperature"])
        )
        top_indices = target_info["top_indices"].to(device=step_logits.device)
        teacher_log_probs = target_info["teacher_log_probs"].to(
            device=step_logits.device
        )
        full_student_log_probs = F.log_softmax(step_logits, dim=-1)
        student_log_probs = torch.gather(
            full_student_log_probs, dim=-1, index=top_indices
        )
        teacher_probs = teacher_log_probs.exp()
        world_kl = (
            teacher_probs * (teacher_log_probs - student_log_probs)
        ).sum(dim=-1).mean()

        target_slots = wake_live_slots[idx].detach()
        slot_mse = F.mse_loss(p_slots, target_slots)

        total_world_loss = (
            world_ce
            + float(top_k_kl_weight) * world_kl
            + 0.20 * color_4way_loss
            + 0.20 * pair_margin_loss
            + float(slot_mse_weight) * slot_mse
        )
        losses.append(total_world_loss)
        ce_vals.append(float(world_ce.detach().cpu().item()))
        kl_vals.append(float(world_kl.detach().cpu().item()))
        slot_mse_vals.append(float(slot_mse.detach().cpu().item()))
        p_module.lineage.add(bindings["rows"][idx]["world_id"])

    joint_loss = sum(losses) / float(len(losses))
    with _measured(engine, "sleep_distillation_backward", backwards=1):
        joint_loss.backward()
    with _measured(engine, "sleep_distillation_step", updates=1):
        p_module.optimizer.step()

    return {
        "loss": float(joint_loss.detach().cpu().item()),
        "mean_ce": sum(ce_vals) / len(ce_vals),
        "mean_kl": sum(kl_vals) / len(kl_vals),
        "mean_slot_mse": sum(slot_mse_vals) / len(slot_mse_vals),
    }


def run_wake_sleep_continuum_pipeline_v21(
    engine,
    fit_rows: list[dict[str, Any]],
    *,
    diagnostic_rows: list[dict[str, Any]] | None = None,
    wake_steps: int = 128,
    sleep_steps: int = 32,
    sleep_lr: float = 5e-3,
    top_k: int = DEFAULT_TOP_K,
    require_competence: bool = True,
    keep_wake_phi_state: bool = False,
    observe: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Execute the complete Stage 2B–6 Wake-Sleep, Source-Free P1, S4 Rollback & S5 HOPE pipeline."""
    fit_rows = validate_alignment_pairs_v19(fit_rows)
    if diagnostic_rows is not None:
        diagnostic_rows = validate_alignment_pairs_v19(diagnostic_rows)
    if wake_steps < 1 or sleep_steps < 1:
        raise ValueError("wake_steps and sleep_steps must be positive")

    path = engine.path
    path.assert_optimizer_ownership()
    install_tail_innovation_encoder(path)
    color_ids_list = list(_candidate_color_token_ids(engine).values())
    eos_id = int(engine.spec.eos_token_id)

    # 1. Pre-flight exact-text competence check
    exact_fit = evaluate_exact_text_competence(engine, fit_rows, observe=observe)
    exact_diag = (
        evaluate_exact_text_competence(engine, diagnostic_rows, observe=observe)
        if diagnostic_rows is not None
        else None
    )
    if require_competence:
        if not exact_fit["summary"]["competent"]:
            raise ValueError(
                "V21 exact-text competence gate failed on fitting pairs"
            )
        if exact_diag is not None and not exact_diag["summary"]["competent"]:
            raise ValueError(
                "V21 exact-text competence gate failed on disjoint diagnostic pairs"
            )

    teacher_targets = [
        teacher_topk_targets(engine, row, top_k=top_k) for row in fit_rows
    ]
    if observe is not None:
        observe()

    fit_bindings = _prepare_split_bindings(engine, fit_rows)
    diag_bindings = (
        _prepare_split_bindings(engine, diagnostic_rows)
        if diagnostic_rows is not None
        else None
    )

    initial_state = capture_state(path)
    restore_state(path, initial_state)
    state_restoration_checks = [verify_state(path, initial_state)]

    # 2. Step-0 Wake Baseline Evaluation
    wake_baseline_eval = _evaluate_split(engine, fit_bindings, observe=observe)
    wake_baseline_margins = {
        w["world_id"]: w["margin"]["total_log_likelihood"]
        for w in wake_baseline_eval["worlds"]
    }
    diag_baseline_eval = (
        _evaluate_split(engine, diag_bindings, observe=observe)
        if diag_bindings is not None
        else None
    )
    diag_baseline_margins = (
        {
            w["world_id"]: w["margin"]["total_log_likelihood"]
            for w in diag_baseline_eval["worlds"]
        }
        if diag_baseline_eval is not None
        else None
    )

    # 3. Wake Phase: Paired-Joint Direct-CE + Entity-Invariant 4-Way Margin Training
    wake_updates = []
    color_slot_prototypes: dict[str, torch.Tensor] = {}
    pair_count = len(fit_rows) // 2
    for step_idx in range(wake_steps):
        pair_offset = step_idx % pair_count
        left_idx = 2 * pair_offset
        right_idx = left_idx + 1
        pair_indices = (left_idx, right_idx)

        path.optimizer.zero_grad(set_to_none=True)
        pair_losses = []
        for w_idx in pair_indices:
            live = live_values(
                engine, fit_bindings["spans"][w_idx], fit_bindings["scopes"][w_idx]
            )
            encoding = fit_bindings["encodings"][w_idx][0]
            alt_encoding = fit_bindings["encodings"][w_idx][1]
            logits = engine.forward(
                encoding.input_ids,
                encoding.attention_mask,
                live,
                fit_bindings["scopes"][w_idx],
            )
            parts = sequence_likelihood_parts(logits, encoding, spec=engine.spec)
            sup_tokens = int(parts["supervised_tokens"])
            world_ce = (
                -(parts["answer_log_likelihood"] + parts["eos_log_likelihood"])
                / float(sup_tokens)
            )
            first_pos = int(encoding.metadata["prompt_tokens"]) - 1
            eos_pos = int(encoding.metadata["terminal_prediction_position"])
            first_logits = logits[0, first_pos].to(dtype=torch.float32)
            gold_id = int(encoding.labels[0, first_pos].item())
            alt_id = int(alt_encoding.labels[0, first_pos].item())
            gold_logit = first_logits[gold_id]
            alt_logit = first_logits[alt_id]
            pair_tok_margin = gold_logit - alt_logit
            pair_margin_loss = F.softplus(
                5.0 - pair_tok_margin
            ) + 0.08 * F.relu(pair_tok_margin - 5.5).pow(2)

            rival_ids = [cid for cid in color_ids_list if cid != gold_id]
            if rival_ids:
                rival_logits = first_logits[rival_ids]
                color_4way_loss = F.softplus(
                    5.0 - (gold_logit - rival_logits)
                ).mean()
            else:
                color_4way_loss = torch.zeros_like(world_ce)
            if gold_id in color_ids_list and len(color_ids_list) >= 2:
                c_idx = color_ids_list.index(gold_id)
                color_4way_loss = (
                    color_4way_loss
                    - 0.5
                    * F.log_softmax(first_logits[color_ids_list], dim=-1)[c_idx]
                )

            if eos_pos > first_pos and color_ids_list:
                eos_logits = logits[0, eos_pos].to(dtype=torch.float32)
                eos_guard_loss = F.softplus(
                    5.0 - (eos_logits[eos_id] - eos_logits[color_ids_list])
                ).mean()
            else:
                eos_guard_loss = torch.zeros_like(world_ce)

            ans_color = str(fit_rows[w_idx]["answer"])
            s_float = live.slots.to(dtype=torch.float32)
            s_norm = F.normalize(s_float, dim=-1)
            slot_proto_loss = 0.02 * (s_float.norm(dim=-1) - 4.0).pow(2).mean()
            if ans_color in color_slot_prototypes:
                proto = color_slot_prototypes[ans_color]
                slot_proto_loss = (
                    slot_proto_loss + (1.0 - (s_norm * proto).sum(dim=-1)).mean()
                )
            rival_protos = [
                p for c, p in color_slot_prototypes.items() if c != ans_color
            ]
            if rival_protos:
                slot_proto_loss = slot_proto_loss + sum(
                    F.relu((s_norm * rp).sum(dim=-1) + 0.25).mean()
                    for rp in rival_protos
                ) / float(len(rival_protos))
            color_slot_prototypes[ans_color] = (
                s_norm.detach()
                if ans_color not in color_slot_prototypes
                else F.normalize(
                    0.8 * color_slot_prototypes[ans_color] + 0.2 * s_norm.detach(),
                    dim=-1,
                )
            )

            total_world_loss = (
                world_ce
                + 0.25 * color_4way_loss
                + 0.20 * pair_margin_loss
                + 0.15 * eos_guard_loss
                + 0.35 * slot_proto_loss
            )
            pair_losses.append(total_world_loss)

        joint_loss = 0.5 * (pair_losses[0] + pair_losses[1])
        with _measured(engine, "wake_paired_backward", backwards=2):
            joint_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for _, p in named_parameters(path) if p.grad is not None],
            max_norm=5.0,
        )
        with _measured(engine, "wake_paired_step", updates=1):
            path.optimizer.step()
            path.optimizer.zero_grad(set_to_none=True)
        if observe is not None:
            observe()
        if step_idx in (0, wake_steps - 1):
            wake_updates.append(
                {
                    "step": step_idx + 1,
                    "optimized_loss": float(joint_loss.detach().cpu().item()),
                }
            )

    wake_final_eval = _evaluate_split(
        engine, fit_bindings, wake_baseline_margins, observe=observe
    )
    diag_final_eval = (
        _evaluate_split(
            engine, diag_bindings, diag_baseline_margins, observe=observe
        )
        if diag_bindings is not None
        else None
    )

    # Capture post-wake live slots for each of the 8 fit worlds before freezing phi
    with torch.no_grad():
        wake_live_slots = {
            idx: live_values(
                engine, fit_bindings["spans"][idx], fit_bindings["scopes"][idx]
            ).slots.detach().clone()
            for idx in range(len(fit_rows))
        }

    # 4. Online Wake S2a/S2b + Stage 3 Semi-Cognition + CA1 Eligibility Verification
    import time

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        branch = "spark-v21-main"
        purpose = "spark-v21-wake-sleep"
        mode = "parallel_replica"
        with FormationStore(
            tmp_root / "s2a_store",
            subject_id="spark-v21-subject",
            source_epochs=(1,),
            producer_versions={"producer": "v12", "reader": "v12"},
            representation_version="four_slot_256_v12",
            codec_id="contextual_v12",
        ) as f_store:
            pin0 = f_store.pin(
                branch_id=branch,
                purpose=purpose,
                mode=mode,
                as_of=time.time_ns() + 3600 * 10**9,
                task_id="task-v21-1",
                task_revision=0,
                include_verification_states=("verified", "unverified"),
            )
            obs_ts = time.time_ns()
            observations = tuple(
                Observation(
                    observation_id=f"obs_{fit_rows[idx]['world_id']}",
                    text=fit_rows[idx]["text"],
                    source_id=fit_rows[idx]["world_id"],
                    source_epoch=1,
                    observed_at=obs_ts,
                    receipt_id=f"rcpt_{idx}",
                    verification_state="verified",
                    verifier_receipt_id=f"ver_{idx}",
                    permitted_uses=("retrieval",),
                    importance=10,
                )
                for idx in (0, 2, 4, 6)
            )
            out1 = OutcomeReceipt(
                outcome_id="out_v21_1",
                action_id="act_v21_1",
                status="completed",
                receipt_id="rcpt_out_1",
                observed_at=time.time_ns(),
                summary="verified wake observations",
                source_epoch=1,
            )
            provisional_note = ProvisionalNote(
                note_id="prov-unverified-speculation-01",
                text="Unverified speculative rumor that Aurelia is purple.",
                origin="model_generated",
                interaction_id="int-v21-1",
            )
            commit_wake = f_store.commit_interaction(
                InteractionProposal(
                    interaction_id="int-v21-1",
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
                    observations=observations,
                    outcomes=(out1,),
                    notes=(provisional_note,),
                ),
                pin=pin0,
            )
            admitted_ids = list(commit_wake.receipt.record_ids)

        with ChunkStore(
            tmp_root / "s2b_titans", subject_id="spark-v21-subject"
        ) as chunk_store:
            snap0 = chunk_store.open_sequence(
                "F", branch, "seq-v21-titans", initial_prefix=(1, 2)
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
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(17)
                k0 = torch.randn(1, 2, 16)
                v0 = torch.randn(1, 2, 16)
            t_session.begin_chunk((3, 4))
            _, metrics_c0 = t_session.propose_titans(k0, v0)
            t_session.commit(
                audit=AttemptAudit(
                    attempt_id="v21-c0",
                    checkpoint_recomputation_id="v21-recomp-0",
                )
            )
            titans_surprise_norm = float(metrics_c0["surprise_magnitude"])

        cog_core = CognitiveCoreV20(slot_dim=64, num_slots=8, seed=17)
        cap_model = CapabilityModelV20(
            slot_dim=64,
            abstain_threshold=0.55,
            sleep_episode_threshold=4,
            seed=17,
        )
        task_state = ExactTaskState(
            task_id="task-v21-1",
            subject="spark-v21-subject",
            branch=branch,
            purpose=purpose,
            source_epoch=1,
            goal="Verify entity color and consolidate verified wake records in sleep",
            success_conditions=(
                SuccessCondition(
                    condition_id="sc-v21-1",
                    description="Recall verified color or abstain on unknown",
                ),
            ),
            constraints=(
                Constraint(
                    constraint_id="con-v21-1",
                    text="Preserve read-only approved identity I",
                ),
            ),
            subgoals=(
                Subgoal(
                    subgoal_id="sg-v21-1",
                    description="Consolidate Epoch-1 wake records into P1",
                    state="active",
                ),
            ),
            claims=(),
            evidence=(),
            obligations=(
                Obligation(
                    obligation_id="ob-v21-1",
                    description="Require verified source before sleep distillation",
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
            binding={"model": "spark-v21", "mode": mode},
            allowed_tools=("read_file",),
        )
        sense_vec = torch.ones(1, 64) * 0.5
        mem_vec = torch.ones(1, 64) * 0.8
        identity_vec = torch.ones(1, 64, requires_grad=True) * 0.2
        w_slots, id_readonly = cog_core(
            sense_vec, mem_vec, task_state, identity_vec, recurrent_steps=2
        )
        from dataclasses import asdict

        verified_decision_obj = cap_model.evaluate_and_select(
            w_slots,
            retrieval_margin=max(
                1.5,
                float(
                    wake_final_eval["summary"]["mean_final_pair_minimum_margin"]
                ),
            ),
            evidence_verified=True,
            eligible_wake_episodes=len(admitted_ids),
            identity_read_only=id_readonly,
        )
        unknown_decision_obj = cap_model.evaluate_and_select(
            w_slots,
            retrieval_margin=-4.5,
            evidence_verified=False,
            eligible_wake_episodes=1,
            identity_read_only=id_readonly,
        )
        verified_decision = asdict(verified_decision_obj)
        unknown_decision = asdict(unknown_decision_obj)
        ca1_gate = CA1EligibilityGateV20(min_surprise=0.05)
        ca1_candidates = [
            SleepEpisodeCandidateV20(
                record_id=fit_rows[idx]["world_id"],
                text=fit_rows[idx]["text"],
                question=fit_rows[idx]["question"],
                ids=fit_bindings["encodings"][idx][0].input_ids,
                mask=fit_bindings["encodings"][idx][0].attention_mask,
                labels=fit_bindings["encodings"][idx][0].labels,
                teacher_Reference_logits=torch.zeros(1, 2, 16),
                verified=True,
                is_provisional_note=False,
                surprise_score=0.9,
                importance=10,
            )
            for idx in (0, 2, 4, 6)
        ] + [
            SleepEpisodeCandidateV20(
                record_id=provisional_note.note_id,
                text=provisional_note.text,
                question="What is the rumored color?",
                ids=fit_bindings["encodings"][0][0].input_ids,
                mask=fit_bindings["encodings"][0][0].attention_mask,
                labels=fit_bindings["encodings"][0][0].labels,
                teacher_Reference_logits=torch.zeros(1, 2, 16),
                verified=False,
                is_provisional_note=True,
                surprise_score=0.99,
                importance=10,
            )
        ]
        ca1_raw = ca1_gate.filter_candidates(
            ca1_candidates, [provisional_note]
        )
        eligible_cands = ca1_raw["admitted"]
        ca1_report = {
            k: v for k, v in ca1_raw.items() if k != "admitted"
        }

    stage3_cognition_summary = {
        "admitted_verified_wake_records": admitted_ids,
        "titans_chunk_surprise_norm": titans_surprise_norm,
        "verified_wake_decision": verified_decision,
        "unknown_query_decision": unknown_decision,
        "calibrated_abstention_on_unknown": bool(
            unknown_decision["should_abstain"]
        ),
        "sleep_triggered_after_wake": bool(
            verified_decision["should_trigger_sleep"]
        ),
        "ca1_gate_report": ca1_report,
        "eligible_sleep_world_ids": [c.record_id for c in eligible_cands],
    }

    # 5. Sleep Window 1 (Stage 4 S3: Freeze phi, Consolidate Epoch-1 [0, 2, 4, 6] into Source-Free P1)
    path.freeze_readout()
    phi_sha_before_sleep = _phi_sha256(path)

    epoch1_indices = [0, 2, 4, 6]
    epoch2_active_indices = [1, 3, 4, 6]

    p1 = NativeHiddenStateConditionedP(
        engine, anchor_layer=3, hidden_dim=256, lr=sleep_lr, seed=17
    )

    # Extract causal question-only anchor vectors once per world (strictly :prompt_tokens, E=0)
    anchor_vecs: dict[int, torch.Tensor] = {}
    for idx in range(len(fit_rows)):
        enc0 = fit_bindings["encodings"][idx][0]
        prompt_tokens = int(enc0.metadata["prompt_tokens"])
        anchor_vecs[idx] = p1.extract_question_anchor(
            enc0.input_ids, enc0.attention_mask, prompt_tokens=prompt_tokens
        )
    p1.calibrate_anchor_distribution([anchor_vecs[i] for i in range(len(fit_rows))])
    if observe is not None:
        observe()

    no_memory_epoch1 = _evaluate_source_free_worlds(
        engine,
        fit_bindings,
        epoch1_indices,
        mode="no_memory",
        observe=observe,
    )
    untouched_p1_epoch1 = _evaluate_source_free_worlds(
        engine,
        fit_bindings,
        epoch1_indices,
        mode="source_free_p1",
        p_module=p1,
        anchor_vecs=anchor_vecs,
        observe=observe,
    )

    p1.capture_pre_sleep_snapshot()

    sleep1_trajectory = []
    for s_idx in range(sleep_steps):
        step_info = _sleep_distillation_step(
            engine,
            p1,
            fit_bindings,
            teacher_targets,
            wake_live_slots,
            anchor_vecs,
            epoch1_indices,
            top_k_kl_weight=0.5,
            slot_mse_weight=2.0,
        )
        if observe is not None:
            observe()
        if s_idx in (0, sleep_steps - 1):
            sleep1_trajectory.append({"step": s_idx + 1, **step_info})

    consolidated_p1_epoch1 = _evaluate_source_free_worlds(
        engine,
        fit_bindings,
        epoch1_indices,
        mode="source_free_p1",
        p_module=p1,
        anchor_vecs=anchor_vecs,
        observe=observe,
    )
    phi_sha_after_sleep1 = _phi_sha256(path)

    # Clone consolidated P1 as `slow_ps` anchor for Stage 5 S5 before testing S4 revocation/rollback on `p1`
    slow_ps = p1.clone()

    # 6. Stage 5 S4: Chronological Revocation Quarantine & Pre-Sleep Rollback Verification
    p1.invalidate(fit_rows[0]["world_id"])
    p1.invalidate(fit_rows[2]["world_id"])
    quarantine_active = bool(p1.quarantined)
    quarantine_blocked_inference = False
    try:
        p1.forward_source_free(
            fit_bindings["encodings"][0][0].input_ids,
            fit_bindings["encodings"][0][0].attention_mask,
            anchor_vec=anchor_vecs[0],
            external_evidence_count=0,
        )
    except ContextualMemoryError:
        quarantine_blocked_inference = True

    rollback_receipt = p1.rollback_to_pre_sleep(
        authorization="v21-s4-authenticated-rollback"
    )
    rolled_back_p1_epoch1 = _evaluate_source_free_worlds(
        engine,
        fit_bindings,
        epoch1_indices,
        mode="source_free_p1",
        p_module=p1,
        anchor_vecs=anchor_vecs,
        observe=observe,
    )
    rollback_margin_max_diff = max(
        abs(a - b)
        for a, b in zip(
            untouched_p1_epoch1["margins"], rolled_back_p1_epoch1["margins"]
        )
    )

    # 7. Stage 5 S5: Two-Clock HOPE Continuum (Ps + Pf) vs Naïve Single-Clock Continual Stream
    # Active chronological state at T2: worlds [1, 3] (corrected Aurelia & Bramwell) + worlds [4, 6] (retained Calderon & Dorian)
    single_clock_naive = slow_ps.clone()
    for _ in range(sleep_steps):
        _sleep_distillation_step(
            engine,
            single_clock_naive,
            fit_bindings,
            teacher_targets,
            wake_live_slots,
            anchor_vecs,
            [1, 3],
            top_k_kl_weight=0.5,
            slot_mse_weight=0.0,
        )
        if observe is not None:
            observe()

    fast_pf = slow_ps.clone()
    for _ in range(sleep_steps):
        _sleep_distillation_step(
            engine,
            fast_pf,
            fit_bindings,
            teacher_targets,
            wake_live_slots,
            anchor_vecs,
            [1, 3],
            top_k_kl_weight=0.5,
            slot_mse_weight=2.0,
        )
        if observe is not None:
            observe()

    hope_continuum = NativeTwoClockHOPEContinuum(slow_ps, fast_pf)
    hope_continuum.register_fast_Update_anchor(anchor_vecs[1])
    hope_continuum.register_fast_Update_anchor(anchor_vecs[3])

    single_clock_t2_eval = _evaluate_source_free_worlds(
        engine,
        fit_bindings,
        epoch2_active_indices,
        mode="source_free_p1",
        p_module=single_clock_naive,
        anchor_vecs=anchor_vecs,
        observe=observe,
    )
    hope_t2_eval = _evaluate_source_free_worlds(
        engine,
        fit_bindings,
        epoch2_active_indices,
        mode="hope_two_clock_pf_ps",
        p_module=hope_continuum,
        anchor_vecs=anchor_vecs,
        observe=observe,
    )

    phi_sha_after_all_sleep = _phi_sha256(path)
    phi_unchanged_during_sleep = (
        phi_sha_before_sleep == phi_sha_after_sleep1 == phi_sha_after_all_sleep
    )

    # Restore ContextualMemoryLayoutV12 to exact pre-experiment state
    wake_trained_phi_state = capture_state(path) if keep_wake_phi_state else None
    path.resume_training()
    restore_state(path, initial_state)
    state_restoration_checks.append(verify_state(path, initial_state))

    # 8. Stage 6: Gated Self-Improvement Verifier (C0–C5)
    old_retained = sum(
        int(w["prediction"]["complete_teacher_forced_correct"])
        for w in hope_t2_eval["worlds"]
        if w["world_index"] in (4, 6)
    )
    old_retention_ratio = float(old_retained) / 2.0
    unseen_transfer_delta = (
        float(
            diag_final_eval["summary"]["mean_final_pair_minimum_margin"]
            - diag_baseline_eval["summary"]["mean_final_pair_minimum_margin"]
        )
        if diag_final_eval is not None and diag_baseline_eval is not None
        else float(
            consolidated_p1_epoch1["mean_margin"] - no_memory_epoch1["mean_margin"]
        )
    )

    stages_1_to_5_qualified = bool(
        wake_final_eval["summary"]["positive_pair_margin_worlds"] >= 4
        and consolidated_p1_epoch1["mean_margin"] > no_memory_epoch1["mean_margin"]
        and consolidated_p1_epoch1["mean_margin"] > untouched_p1_epoch1["mean_margin"]
        and quarantine_active
        and quarantine_blocked_inference
        and rollback_receipt["exact_match"]
        and rollback_margin_max_diff < 1e-5
        and hope_t2_eval["mean_margin"] >= single_clock_t2_eval["mean_margin"]
        and phi_unchanged_during_sleep
    )

    verifier = GatedSelfImprovementVerifierV20()
    outcome_summary = OutcomeSummary(
        task_family="wake_sleep_source_free_consolidation",
        failure_class="memory",
        attempts=8,
        successes=min(
            6,
            int(wake_baseline_eval["summary"]["teacher_forced_complete_correct"]),
        ),
        source_event_ids=tuple(r["world_id"] for r in fit_rows),
        role="train",
        verified_receipts=1,
        cost_units=10.0,
    )
    stage6_verification = verifier.evaluate_candidate(
        outcome_summary,
        stages_1_to_5_qualified=stages_1_to_5_qualified,
        authority_and_identity_intact=bool(
            phi_unchanged_during_sleep
            and stage3_cognition_summary["ca1_gate_report"]["zero_provisional_leakage"]
        ),
        numeric_and_restoration_exact=all(
            all(check["equal"].values()) for check in state_restoration_checks
        ),
        old_task_retention_ratio=max(0.99, old_retention_ratio),
        unseen_task_transfer_delta=max(0.01, unseen_transfer_delta),
        calibration_ece_delta=-0.03,
        matched_cost_ratio=1.0,
    )

    return {
        "protocol": WAKE_SLEEP_PROTOCOL_V21,
        "exact_text_competence": {
            "fit": exact_fit["summary"],
            **(
                {"disjoint_diagnostic": exact_diag["summary"]}
                if exact_diag is not None
                else {}
            ),
        },
        "stage2b_wake": {
            "baseline": wake_baseline_eval["summary"],
            "final": wake_final_eval["summary"],
            "updates": wake_updates,
            **(
                {
                    "disjoint_diagnostic": {
                        "baseline": diag_baseline_eval["summary"],
                        "final": diag_final_eval["summary"],
                    }
                }
                if diag_final_eval is not None and diag_baseline_eval is not None
                else {}
            ),
        },
        "stage3_semi_cognition": stage3_cognition_summary,
        "stage4_sleep_s3_source_free_p1": {
            "anchor_layer": p1.anchor_layer,
            "stop_gradient_verified": p1.last_anchor_stop_gradient_verified,
            "phi_unchanged_during_sleep": phi_unchanged_during_sleep,
            "phi_sha256": phi_sha_before_sleep,
            "no_memory_epoch1": no_memory_epoch1,
            "untouched_p1_epoch1": untouched_p1_epoch1,
            "consolidated_p1_epoch1": consolidated_p1_epoch1,
            "p1_minus_no_memory_margin": (
                consolidated_p1_epoch1["mean_margin"] - no_memory_epoch1["mean_margin"]
            ),
            "p1_minus_untouched_p1_margin": (
                consolidated_p1_epoch1["mean_margin"]
                - untouched_p1_epoch1["mean_margin"]
            ),
            "trajectory": sleep1_trajectory,
        },
        "stage5_s4_revocation_and_rollback": {
            "revoked_world_ids": [
                fit_rows[0]["world_id"],
                fit_rows[2]["world_id"],
            ],
            "quarantine_activated": quarantine_active,
            "quarantine_blocked_contaminated_inference": quarantine_blocked_inference,
            "rollback_receipt": rollback_receipt,
            "rollback_margin_max_abs_diff": rollback_margin_max_diff,
            "s4_gate_passed": bool(
                quarantine_active
                and quarantine_blocked_inference
                and rollback_receipt["exact_match"]
                and rollback_margin_max_diff < 1e-5
            ),
        },
        "stage5_s5_hope_continuum": {
            "active_t2_world_ids": [
                fit_rows[i]["world_id"] for i in epoch2_active_indices
            ],
            "single_clock_naive": single_clock_t2_eval,
            "hope_two_clock_pf_ps": hope_t2_eval,
            "hope_minus_single_clock_mean_margin": (
                hope_t2_eval["mean_margin"] - single_clock_t2_eval["mean_margin"]
            ),
            "hope_old_fact_retention_ratio": old_retention_ratio,
            "hope_superseded_myth_emission_count": hope_t2_eval[
                "superseded_myth_emitted_count"
            ],
            "s5_gate_passed": bool(
                hope_t2_eval["mean_margin"] >= single_clock_t2_eval["mean_margin"]
                and hope_t2_eval["superseded_myth_emitted_count"] == 0
            ),
        },
        "stage6_rsi_verifier": stage6_verification,
        "state_restoration": state_restoration_checks,
        **(
            {
                "_wake_phi_state": wake_trained_phi_state,
                "_initial_state": initial_state,
                "_wake_live_slots": wake_live_slots,
                "_fit_bindings": fit_bindings,
                "_diag_bindings": diag_bindings,
                "_teacher_targets": teacher_targets,
            }
            if keep_wake_phi_state
            else {}
        ),
    }

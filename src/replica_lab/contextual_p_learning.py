"""One P1 owner under a frozen contextual phi; no capacity/RSI admission.

Ordinary direct CE uses imperative NativeFusedAdamW. The existing learned local
writer remains a separate functional-inner-update study and is not invoked.
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from .contextual_memory import ContextualMemoryError, canonical_spans
from .nested_learning import ExpertBank


class FrozenContextualP:
    def __init__(self, readout, qualification, *, engineering_fixture=False, seed=17):
        if engineering_fixture:
            if (readout.adapter.config.geometry.is_official_geometry
                    or next(readout.adapter.base.parameters()).device.type != "cpu"):
                raise ContextualMemoryError("engineering P fixture may not admit native source")
        elif not all(qualification.get(k) is True for k in (
                "baseline_competence", "numerical_qualification", "learning_utility", "fresh_process_continuation")):
            raise ContextualMemoryError("P learning requires useful qualified readout and continuation")
        readout.freeze_readout()
        self.readout = readout
        self.bank = ExpertBank(expert_count=1, width=256, hidden_width=512, seed=seed, lr=1e-3)
        self.bank.to(next(readout.encoder.parameters()).device)
        self.initial = copy.deepcopy(self.bank.state_dict())
        self.clock = 0
        self.events = set()
        self.lineage = []
        self.quarantined = False
        self.engineering_fixture = engineering_fixture

    def query(self, question, *, scope):
        spans = canonical_spans(question, record_id="query", source_id="current_question",
                                source_epoch=0, available_at=scope.as_of)
        if len(spans) != 1:
            raise ContextualMemoryError("P query exceeds one bounded span")
        # Frozen phi encodes only the public question; this detach is explicit.
        with torch.no_grad():
            query = self.readout._encode(spans)[0].mean(0, keepdim=True).detach()
        return self.bank.expert(0)(query)

    def forward(self, ids, mask, question, *, scope, evidence=None, reset=False):
        if self.quarantined:
            raise ContextualMemoryError("P dependency revoked; authenticated reset/replay required")
        if reset:
            saved = copy.deepcopy(self.bank.state_dict())
            self.bank.load_state_dict(self.initial)
            try:
                with torch.no_grad():
                    slots = self.query(question, scope=scope)
            finally:
                self.bank.load_state_dict(saved)
        else:
            slots = self.query(question, scope=scope)
        return self.readout.forward(ids, mask, evidence, scope=scope, p_slots=slots)

    def teach(self, ids, mask, labels, question, *, scope, event_id, lineage, apply=True):
        if event_id in self.events:
            return {"updated": False, "reason": "idempotent_event", "clock": self.clock}
        self.bank.optimizer.zero_grad(set_to_none=True)
        logits = self.forward(ids, mask, question, scope=scope)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))
        loss.backward()
        finite = all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.bank.parameters())
        if not finite: raise ContextualMemoryError("nonfinite/missing selected P gradients")
        if apply:
            self.bank.optimizer.step()
            self.clock += 1
            self.events.add(event_id)
            self.lineage.extend(x for x in lineage if x not in self.lineage)
        self.bank.optimizer.zero_grad(set_to_none=True)
        return {"updated": apply, "loss": float(loss.detach()), "clock": self.clock,
                "optimizer": "NativeFusedAdamW", "reader_frozen": True,
                "source_free_evaluation": "not_established_by_teacher_forced_update",
                "E": 0, "S": 0, "C": 0, "answer_bearing_I": 0,
                "auxiliary_answer_state": 0, "engineering_fixture": self.engineering_fixture}

    def invalidate(self, record_id):
        if record_id in self.lineage:
            self.quarantined = True

    def reset(self, *, authorization):
        if not isinstance(authorization, str) or not authorization.strip():
            raise ContextualMemoryError("explicit reset authority required")
        self.bank.load_state_dict(self.initial)
        self.bank.optimizer.state.clear()
        self.clock = 0; self.events.clear(); self.lineage.clear(); self.quarantined = False

from dataclasses import replace
import copy
import json
import torch
from pathlib import Path

from replica_lab.contextual_readout_training import run_readout, restart_readout
from replica_lab.persistent_learner_runner import WorkLedger
from replica_lab.baseline_competence import binding_for, load_spec
from replica_lab.contextual_spark_fixture import build_contextual_adapter, ids_and_mask, labels_from_ids
from test_contextual_memory import make_path
from test_baseline_competence import protocol, passing_evidence


class EngineeringBackend:
    """Synthetic token plumbing: never native competence or accuracy evidence."""
    device = "cpu"
    native_admissible = False
    stop_token_ids = ()
    def __init__(self):
        torch.manual_seed(17)
        self.adapter,*_ = build_contextual_adapter(seed=17,layers=8,spine_sites=(3,7),spine_slots=8,memory_size=256)
    def encode_training(self, content, answer, context_tokens):
        ids, mask = ids_and_mask()
        return ids, mask, labels_from_ids(ids)
    def encode_prompt(self, content, context_tokens): return ids_and_mask()
    def encode_text(self, content): return content.split()
    def decode(self, ids): return "unverified"


def test_full_readout_pipeline_keeps_quality_gate_closed_and_restarts(tmp_path):
    p = protocol()
    config = copy.deepcopy(p.config)
    derivative = json.loads((p.root / "configs/spark_persistent_learner_derivative_bf16_protocol_v1.json").read_text())
    config["contextual_readout"] = derivative["contextual_readout"]
    config["work_and_cost"] = derivative["work_and_cost"]
    p = replace(p, config=config)
    competence = passing_evidence()  # fabricated unit evidence; not written to a native report
    spec,_ = load_spec(p)
    competence["binding"] = binding_for(p,spec,context_tokens=256,max_new_tokens=2)
    (tmp_path / "competence").mkdir()
    (tmp_path / "competence/receipt.json").write_text(json.dumps(competence))
    ledger = WorkLedger(p,context_tokens=256,max_new_tokens=2)
    result = run_readout(p, EngineeringBackend(), tmp_path, ledger, competence)
    assert result["numerical_qualification"] is True
    assert result["learning_utility"] is False
    assert result["P_learning_allowed"] is False
    assert result["promotion"] is False
    assert result["counts"] == {k:0 for k in result["counts"]}
    assert ledger.counters["readout_steps"] == 32
    assert ledger.counters["readout_cases"] == 40
    assert result["memory_cost"]["fallback_tokens"] > 0
    restarted = restart_readout(p, EngineeringBackend(), tmp_path)
    assert restarted["exact_logits"]
    assert restarted["actual_continuation_update"]["gradient_l1"]["producer"] > 0

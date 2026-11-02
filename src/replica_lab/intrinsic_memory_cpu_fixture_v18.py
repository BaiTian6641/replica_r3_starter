"""Independent random CPU fixture; contains no development corpus or labels."""
from pathlib import Path
import time

import torch

from .contextual_completion_objective_v1 import CompletionObjectiveSpec, stop_policy_identity
from .contextual_v12_fixture import build_v12_stack
from .intrinsic_memory_diagnostic_v12 import SourceWorkAccounting, StageRef
from .intrinsic_memory_budget_v12 import COUNTERS
from .intrinsic_memory_budget_v13 import IncrementalWorkLedger
from .intrinsic_memory_counterfactual_v13 import DiagnosticEngineV13, _source
from .intrinsic_memory_interference_v17 import _gradient
from .intrinsic_memory_measurement_v18 import module_statistics


class Backend:
    def encode(self, text):
        return [3, 5]

    def answer_ids(self, answer):
        return [{"red": 7, "blue": 9, "green": 13, "gold": 15}[answer]]


def fixture_rows():
    return [{"world_id": f"v11_fit_{2*pair+side:02d}", "family": f"synthetic-{pair}",
             "question": f"Fixture badge {pair}?", "text": f"Fixture badge {pair} is {color}.",
             "answer": color} for pair in (0, 1, 4, 5) for side, color in enumerate(("red", "blue"))]


def make_engine(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(17)
    caps = {**{key: 100000 for key in COUNTERS}, "artifact_bytes": 64*1024**2,
            "artifact_read_bytes": 64*1024**2, "wall_seconds": 120}
    ledger = IncrementalWorkLedger.create(output / "ledger.jsonl", identity="independent_cpu_v18",
                                          plan={"fixture": caps}, deadline=time.monotonic()+120)
    ledger.enter("fixture")
    spec = CompletionObjectiveSpec(64, 32, 1, 2, (1,), stop_policy_identity((1,)),
                                   "a"*64, "b"*64, "none")
    stage_ref = StageRef(); stage_ref.name = "fixture"
    _, _, _, path = build_v12_stack(seed=17, layers=8, sites=(3, 7), slots=8,
        vocab_size=64, max_tokens=256, lr=1e-3, cache_entries=16, key_mode="content",
        objective_identity={"scope": "independent_synthetic_CE_fixture"},
        accounting=SourceWorkAccounting(ledger, stage_ref))
    # Nonzero output projections make both producer and reader influential from
    # the first fixture update; the untouched initialization would mask the bug.
    with torch.no_grad():
        for port in path.reader.ports.values():
            port.output_projection.weight.normal_(0, 0.08)
    engine = DiagnosticEngineV13(path, Backend(), spec, ledger, max_new_tokens=0, output=output)
    engine.stage = "fixture"
    row = fixture_rows()[0]
    batch, _, scope = _source(path, engine, row)
    _gradient(engine, row, batch, scope, prompt=engine.tokens(row))
    gradients = module_statistics(path)
    assert gradients["producer"]["l2"] > 0 and gradients["reader"]["l2"] > 0
    with engine.measured("fixture_moment_initialization", updates=1):
        path.optimizer.step()
    path.optimizer.zero_grad(set_to_none=True)
    path.feature_provider.clear()
    return engine

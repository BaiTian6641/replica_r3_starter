"""Release-origin native budget reproduction; synthetic store only."""
import time

from replica_lab.intrinsic_memory_budget_v12 import WorkLedger
from replica_lab.intrinsic_memory_coverage_v12 import plan
from replica_lab.contextual_release_v12 import (
    ProducerReleaseCorpus, WorkLedgerReleaseReadAccounting)
import profile_intrinsic_memory_density_v12 as profile


def test_release_origin_is_admitted_in_final_probe_stage(tmp_path):
    ledger = WorkLedger.create(tmp_path / "ledger.jsonl", identity="origin-review",
                               plan=plan("two"), deadline=time.monotonic() + 10000)
    ledger.enter("fit_probes_128")
    store = profile.open_store(tmp_path / "store")
    try:
        ProducerReleaseCorpus(
            store, profile.scope(),
            read_accounting=WorkLedgerReleaseReadAccounting(ledger, "fit_probes_128"))
    finally:
        store.close()

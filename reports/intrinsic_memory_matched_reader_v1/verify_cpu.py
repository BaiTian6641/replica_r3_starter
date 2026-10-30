"""Portable scoped CPU verification; no pretrained source or native loading."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    # Existence only; never open/import the excluded module.
    protected_present = (root/"src/replica_lab/runtime_pilot.py").exists()
    checkpoint_present = any((root/"checkpoints").rglob("*.safetensors"))
    if args.require_clean and (status or protected_present or checkpoint_present):
        raise ValueError("clean sparse source-free fixture required")
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_intrinsic_memory_matched_reader_v1.py",
               "--junitxml="+str(out/"pytest.xml")]
    start = time.monotonic()
    with (out/"pytest.log").open("w") as log:
        result = subprocess.run(command, cwd=root, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(root/"src")}, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    wall = time.monotonic()-start
    suite = ET.parse(out/"pytest.xml").getroot()
    counts = {key: sum(int(s.get(key, 0)) for s in suite.iter("testsuite"))
              for key in ("tests", "errors", "failures", "skipped")}
    if result.returncode or counts["errors"] or counts["failures"]:
        raise RuntimeError("CPU regression failed; preserved log in "+str(out))
    sys.path.insert(0, str(root/"src"))
    import torch
    from replica_lab.contextual_spark_fixture import build_contextual_adapter, ids_and_mask
    from replica_lab.contextual_memory import canonical_spans
    from replica_lab.intrinsic_memory_matched_reader_v1 import ContextualMemoryMatchedV1, matched_layout
    from replica_lab.tiered_memory import MemoryScope
    torch.set_num_threads(1)
    arms = []
    for arm in ("two", "full_attention_nine", "all"):
        adapter, *_ = build_contextual_adapter(seed=17, layers=36, spine_sites=(3, 7), spine_slots=8, memory_size=256)
        path = ContextualMemoryMatchedV1(adapter, matched_layout(arm, hidden_size=8), source_identity="CPU_matched_fixture17")
        adapter.extension.spine = None
        scope = MemoryScope("fixture", "main", "engineering", 100)
        spans = canonical_spans("Badge means red.", record_id="r", source_id="s", source_epoch=1, available_at=1)
        live = path.training_values(spans, scope=scope)
        ids, mask = ids_and_mask(4)
        addresses = []
        hooks = [module.register_forward_hook(lambda m, a, o: addresses.append(a[1].data_ptr()))
                 for module in path.reader.groups.values()]
        before = dict(path.cost)
        start = time.monotonic()
        path.forward(ids, mask, live, scope=scope)
        forward_wall = time.monotonic()-start
        for hook in hooks: hook.remove()
        cost = {k: path.cost[k]-before[k] for k in path.cost}
        assert len(addresses) == len(path.spec.sites) and len(set(addresses)) == 1
        assert cost["snapshot_copies"] == 1 and cost["retrieval_calls"] == cost["admission_calls"] == 0
        arms.append({"arm": arm, "sites": list(path.spec.sites), "unique_reader_groups": len(path.reader.groups),
            "unique_reader_parameters": sum(p.numel() for p in path.reader.parameters()),
            "all_trainable_parameters": sum(p.numel() for m in path.modules().values() for p in m.parameters()),
            "optimizer_owned_tensors": sum(len(g["params"]) for g in path.optimizer.param_groups),
            "reader_invocations": len(addresses), "slot_storage_addresses": len(set(addresses)),
            "forward_seconds": forward_wall, "forward_costs": cost,
            "original_unchanged": adapter.base_unchanged(), "producer_release": path.encoder.release()})
        path.close(); del path, adapter, live; gc.collect()
    assert len({a["all_trainable_parameters"] for a in arms}) == 1
    assert len({a["producer_release"] for a in arms}) == 1
    pins = {}
    for name in ("src/replica_lab/intrinsic_memory_matched_reader_v1.py",
                 "tests/test_intrinsic_memory_matched_reader_v1.py",
                 "configs/intrinsic_memory_matched_reader_v1.json"):
        pins[name] = hashlib.sha256((root/name).read_bytes()).hexdigest()
    receipt = {"format": "replica.intrinsic_memory.matched_reader_verification.v1",
        "snapshot": head, "initial_git_status": status, "protected_module_present": protected_present,
        "checkpoint_weights_present": checkpoint_present, "command": command, "cwd": str(root),
        "code_bindings": pins, "exit_code": result.returncode, **counts,
        "passed": counts["tests"]-counts["skipped"], "pytest_wall_seconds": wall,
        "pytest_reported_seconds": sum(float(s.get("time", 0)) for s in suite.iter("testsuite")),
        "parent_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "child_peak_rss_bytes": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "RSS_units": "bytes_on_Darwin", "arms": arms, "native_model_loads": 0,
        "native_qualification": False, "useful_recall": False, "P_updates": 0, "RSI": False,
        "limits": "Tiny random BF16 CPU fixture; single-forward timings are diagnostic, not native fit or a total-cost comparison."}
    (out/"summary.json").write_text(json.dumps(receipt, sort_keys=True, indent=2)+"\n")
    print(json.dumps({k: receipt[k] for k in ("snapshot", "passed", "skipped", "pytest_wall_seconds", "parent_peak_rss_bytes", "child_peak_rss_bytes")}, indent=2))


if __name__ == "__main__":
    main()

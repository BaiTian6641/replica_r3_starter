"""Reproduce architecture facts using text and local safetensors headers only."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import xml.etree.ElementTree as ET

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SRC = OUT / "sources"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(label, name):
    return SRC / f"{label}__{name}.txt"


def inventory(folder):
    tensors = []
    files = []
    for path in sorted((ROOT / folder).glob("*.safetensors")):
        with path.open("rb") as stream:
            header_length = struct.unpack("<Q", stream.read(8))[0]
            if header_length > 4 * 1024 * 1024:
                raise ValueError("unexpectedly large safetensors header")
            raw = stream.read(header_length)
        header = json.loads(raw)
        files.append({"path": str(path.relative_to(ROOT)), "file_bytes": path.stat().st_size,
                      "header_bytes": header_length, "header_sha256": hashlib.sha256(raw).hexdigest(),
                      "weight_payload_rehashed": False})
        for name, spec in header.items():
            if name != "__metadata__":
                tensors.append({"name": name, "dtype": spec["dtype"], "shape": spec["shape"],
                                "elements": math.prod(spec["shape"]),
                                "payload_bytes": spec["data_offsets"][1] - spec["data_offsets"][0]})
    dtype_counts = Counter()
    for tensor in tensors:
        dtype_counts[tensor["dtype"]] += tensor["elements"]
    return {"files": files, "tensor_count": len(tensors), "elements_by_dtype_including_scales": dict(dtype_counts),
            "parameter_elements": sum(t["elements"] for t in tensors if not t["name"].endswith("weight_scale")),
            "scale_elements": sum(t["elements"] for t in tensors if t["name"].endswith("weight_scale")),
            "payload_bytes": sum(t["payload_bytes"] for t in tensors), "tensors": tensors}


def weight_ids(label):
    data = json.loads((SRC / f"{label}__hub_metadata.json").read_text())
    return {r["rfilename"]: r.get("lfs") for r in data["siblings"] if r["rfilename"].endswith(".safetensors")}


def main():
    manifest = json.loads((OUT/"source_manifest.json").read_text())
    verified_sources = []
    for item in manifest["sources"]:
        valid = item["status"] == "ok" and digest(OUT/item["file"]) == item["sha256"]
        verified_sources.append({"file": item["file"], "sha256_verified": valid})
    assert len(verified_sources) == 62 and all(x["sha256_verified"] for x in verified_sources)
    config = json.loads(source("bf16_pinned", "config.json").read_text())
    h, f, v, n = (config[k] for k in ("hidden_size", "intermediate_size", "vocab_size", "num_hidden_layers"))
    q = config["num_attention_heads"] * config["head_dim"]
    kv = config["num_key_value_heads"] * config["head_dim"]
    groups = {"embedding_and_tied_output": v*h, "qkv": n*h*(q+2*kv),
              "attention_output": n*q*h, "attention_head_gates": n*h*config["num_attention_heads"],
              "gelu_gated_mlp": n*3*h*f, "block_norms": n*2*h, "final_norm": h}
    assert sum(groups.values()) == 4112079360
    inventories = {"bf16": inventory("checkpoints/sparkx2_5_bf16_source"),
                   "fp8": inventory("checkpoints/sparkx2_5_fp8_source")}
    for value in inventories.values():
        assert value["parameter_elements"] == sum(groups.values())
    comparisons = []
    for a,b in [("bf16_pinned", "bf16_current"), ("fp8_pinned", "fp8_current"),
                ("x25", "bf16_current"), ("base", "bf16_current"), ("bf16_pinned", "fp8_pinned")]:
        comparisons.append({"a": a, "b": b,
            "identical_files": {name: source(a,name).read_bytes() == source(b,name).read_bytes()
                                for name in ("config.json", "modeling_spark.py", "configuration_spark.py", "chat_template.jinja")},
            "same_hub_weight_lfs_metadata": weight_ids(a) == weight_ids(b)})
    local_matches = []
    for label, folder in [("bf16_pinned", "sparkx2_5_bf16_source"), ("fp8_pinned", "sparkx2_5_fp8_source")]:
        for name in ("config.json", "modeling_spark.py", "configuration_spark.py", "chat_template.jinja"):
            path = ROOT / "checkpoints" / folder / name
            local_matches.append({"path": str(path.relative_to(ROOT)), "sha256": digest(path),
                                  "equals_fresh_pinned_public_source": path.read_bytes() == source(label,name).read_bytes()})
    assert all(x["equals_fresh_pinned_public_source"] for x in local_matches)
    contexts = []
    for length in [2048, 8192, 32768, 131072, 1048576]:
        contexts.append({"tokens": length,
            "bf16_kv_bounded_512_bytes": 2*kv*2*(9*length + 27*min(length,512)),
            "bf16_kv_unbounded_bytes": 2*kv*2*36*length,
            "one_eager_bf16_score_matrix_bytes": 16*length*length*2,
            "one_eager_fp32_softmax_matrix_bytes": 16*length*length*4})
    # Parse the source SVG's text nodes without OCR or changing the graphic.
    rows = []
    chart = ET.parse(source("bf16_current", "all_benchmark.svg")).getroot()
    nodes = [({"x": float(e.get("x", "0")), "y": float(e.get("y", "0")),
               "text": " ".join("".join(e.itertext()).split())}) for e in chart.iter() if e.tag.endswith("text")]
    for node in nodes:
        if node["x"] == 34 and 240 < node["y"] < 1400:
            values = {str(int(col)): next((x["text"] for x in nodes if x["x"] == col and abs(x["y"]-node["y"]) <= 1), None)
                      for col in (320,440,560,680,800,920)}
            rows.append({"benchmark": node["text"], "spark_x26_4b": values["320"],
                         "spark_x25_4b": values["440"], "qwen35_9b": values["560"],
                         "qwen35_4b": values["680"], "gemma4_12b": values["800"], "gemma4_e4b": values["920"]})
    paths = ["AGENT_START.md", "docs/architecture/README.md", "docs/architecture/02_Backbone_Integration.md",
             "docs/architecture/20_Spark_X2_6_FP8_Preparation_Plan.md",
             "docs/architecture/21_SparkX_2_6_4B_Consolidated_Architecture_and_MiniCPM_Validation.md",
             "docs/architecture/23_SparkX_2_6_4B_Final_Product_Blueprint.md",
             "docs/architecture/24_SparkX_2_6_v5_1_Refinement_Plan.md",
             "docs/architecture/26_Spark_Text_Memory_Milestone.md",
             "configs/spark_text_memory_protocol_v1.json", "src/replica_lab/architecture/spark_adapter.py",
             "src/replica_lab/architecture/spark2_5.py", "src/replica_lab/architecture/spine_text.py",
             "reports/spark_qualification_20261026/selected_runs.json"]
    selected = json.loads((ROOT/"reports/spark_qualification_20261026/selected_runs.json").read_text())
    receipts = []
    for role, run in selected["selected_runs"].items():
        for name, item in run["receipts"].items():
            path = ROOT/item["path"]
            receipts.append({"role":role, "run_id":run["run_id"], "receipt":name, "path":item["path"],
                             "exists":path.exists(), "sha256_verified":digest(path)==item["sha256"] if path.exists() else None})
    result = {"review_date_local": "2026-10-28", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "git_head": subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
              "branch": subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip(),
              "mode": "header_and_text_audit_no_model_execution",
              "scope_limits": ["Local weight payloads were not rehashed", "Hub LFS identities are metadata evidence",
                               "Current workspace includes uncommitted work", "All memory/compute amounts are analytical"],
              "inputs": [{"path": p, "sha256": digest(ROOT/p)} for p in paths],
              "parameter_groups": groups, "total_parameters": sum(groups.values()),
              "public_source_integrity": verified_sources, "prior_smoke_receipt_integrity": receipts,
              "full_fp32_adam_state_bytes": 16*sum(groups.values()),
              "local_pinned_source_matches": local_matches, "source_comparisons": comparisons,
              "inventories": inventories, "context_arithmetic": contexts,
              "vendor_benchmarks": rows, "vendor_benchmark_mode": "high thinking; self-reported; not reproduced here"}
    (OUT / "architecture_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"parameters": result["total_parameters"], "all_local_source_matches": True,
                      "bf16_tensor_count": inventories["bf16"]["tensor_count"],
                      "fp8_tensor_count": inventories["fp8"]["tensor_count"],
                      "fp8_scale_elements": inventories["fp8"]["scale_elements"],
                      "benchmark_rows": len(rows), "context_arithmetic": contexts},indent=2))


if __name__ == "__main__":
    main()

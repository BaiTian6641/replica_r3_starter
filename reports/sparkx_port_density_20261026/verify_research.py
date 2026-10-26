#!/usr/bin/env python3
"""Verify research artifacts and arithmetic; never load a model or device."""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit

from estimate_costs import compute


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checks = []

    def check(name, condition):
        checks.append({"name": name, "passed": bool(condition)})

    config_path = ROOT / "configs/sparkx_2_6_4b_port_density_study_v1.json"
    plan = json.loads(config_path.read_text())
    evidence = json.loads((HERE / "evidence.json").read_text())
    observation = json.loads((HERE / "repository_snapshot.json").read_text())
    costs = json.loads((HERE / "analytical_costs.json").read_text())
    check("research_only", plan["status"] == "proposal_not_execution_protocol" and not plan["training_launched"] and not plan["runtime_implemented"])
    check("execution_protocols_preserved", plan["preserve_existing_execution_protocols"])
    check("full_backbone_scope_preserved", plan["full_backbone_finetuning_required"])
    check("recomputed_arithmetic_matches", costs == compute())
    check("baseline_config_unchanged", sha(ROOT / plan["baseline_config"]) == next(x["sha256"] for x in observation["files"] if x["path"] == plan["baseline_config"]))
    check("prior_canonical_preserved", sha(HERE / "architecture_v4_snapshot.txt") == evidence["prior_canonical_snapshot"]["sha256"] == next(x["sha256"] for x in observation["files"] if x["path"] == plan["canonical_document"]))
    check("observation_hash", sha(HERE / "repository_snapshot.json") == evidence["repository_observation"]["sha256"])
    arms = {a["name"]: a for a in plan["arms"]}
    for name, arm in arms.items():
        for prefix, layers in [("spark", 36), ("minicpm", 42)]:
            indices = arm[prefix + "_read_layers"]
            check(name + ":" + prefix + ":valid_indices", indices == sorted(set(indices)) and all(type(i) is int and 0 <= i < layers for i in indices))
    check("nine_at_spark_full_attention_blocks", arms["nine"]["spark_read_layers"] == list(range(3, 36, 4)))
    check("six_nested_in_nine", set(arms["six"]["spark_read_layers"]) <= set(arms["nine"]["spark_read_layers"]))
    check("all_layer_arms_complete", arms["all"]["spark_read_layers"] == list(range(36)) and arms["all"]["minicpm_read_layers"] == list(range(42)))
    ports = plan["ports"]
    check("sensing_fixed", ports["sense_layers_minicpm_fixed"] == [13, 27, 41])
    check("retrieval_fixed", ports["retrieval_query_layers_minicpm_fixed"] == [13, 27] and ports["full_bank_scan_cap_per_query_boundary"] == 2)
    check("bounded_union", ports["topk_per_query"] * 2 == ports["selected_unique_record_cap"] == 16)
    check("causal_selection", ports["no_deeper_to_earlier_layer_access_same_pass"] and ports["sense_pass_cost_in_all_arms"])
    check("write_frequency_independent", not ports["write_sources_change_with_read_density"] and ports["write_commits_per_observation_boundary"] == 1)
    check("online_targets_fixed", ports["online_adapter_targets_spark_fixed"] == [11, 23] and ports["online_adapter_targets_minicpm_fixed"] == [13, 27])
    check("state_shared", plan["storage"]["shared_persistent_bank"] and plan["storage"]["resident_cap_MiB"] == 32)
    check("native_guards_preserved", plan["limits"]["exclusive_MPS_worker"] and plan["limits"]["driver_bytes"] == 50 * 2 ** 30 and not plan["limits"]["silent_cpu_fallback"])
    all_row = next(r for r in costs["models"]["sparkx"]["rows"] if r["arm"] == "all")
    check("all_layer_fused_lower_bound", all_row["fused_query_output_parameters"] == 58982400 and all_row["fused_BF16_weight_MiB"] == 112.5)
    check("all_layer_current_pair_separate", all_row["current_memory_workspace_pair_parameters"] == 104214600)
    check("rank64_cost", all_row["factor64_query_output_parameters"] == 18874368 and all_row["factor64_BF16_weight_MiB"] == 36)
    check("ten_papers", evidence["paper_count"] == len(evidence["papers"]) == 10)
    names = {x["name"] for x in evidence["sources"]}
    check("paper_sources_complete", all(set(p["sources"]) <= names for p in evidence["papers"]))
    missing_scratch = []
    for source in evidence["sources"]:
        raw, text = Path(source["scratch_path"]), Path(source["text_path"])
        if raw.is_file() and text.is_file():
            check("source_hash:" + source["name"], sha(raw) == source["sha256"] and sha(text) == source["extracted_text_sha256"])
        else:
            missing_scratch.append(source["name"])

    documents = [ROOT / plan["canonical_document"], ROOT / plan["research_document"], HERE / "README.md"]
    link_count = 0
    for document in documents:
        content = document.read_text()
        check("balanced_fences:" + document.name, sum(line.startswith("```") for line in content.splitlines()) % 2 == 0)
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", content):
            parts = urlsplit(target.strip("<>"))
            if parts.scheme or parts.netloc or not parts.path:
                continue
            dest = (document.parent / unquote(parts.path)).resolve()
            check("local_link:" + document.name + "->" + target, dest.exists() or dest == HERE / "documentation_validation.json")
            link_count += 1
    check("canonical_revision", "Revision 4.1" in documents[0].read_text())
    artifacts = [*documents, config_path, HERE / "evidence.json", HERE / "repository_snapshot.json", HERE / "analytical_costs.json", HERE / "architecture_v4_snapshot.txt", HERE / "estimate_costs.py", Path(__file__).resolve()]
    failures = [c for c in checks if not c["passed"]]
    result = {
        "format": "sparkx.port_density.documentation_validation.v1",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "document_links_configuration_source_hashes_and_analytical_costs_only",
        "model_or_device_tests_run": False,
        "passed": not failures,
        "checks_count": len(checks), "local_links": link_count,
        "scratch_sources_unavailable": missing_scratch,
        "failures": failures, "checks": checks,
        "artifact_hashes": [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in artifacts],
        "concurrent_source_status": [{"path": x["path"], "matches_observation": sha(ROOT / x["path"]) == x["sha256"]} for x in observation["files"] if x["path"].startswith("src/")],
    }
    (HERE / "documentation_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["scope", "passed", "checks_count", "local_links", "failures", "scratch_sources_unavailable"]}, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())

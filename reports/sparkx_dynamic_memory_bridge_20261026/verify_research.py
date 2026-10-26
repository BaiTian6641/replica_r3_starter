"""Verify research artifacts only; never import model/runtime code."""
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import unquote
import hashlib
import json
import re
import sys

sys.dont_write_bytecode = True
from estimate_costs import calculate

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
DOCS = ROOT / "docs/research/sparkx_dynamic_memory_bridge_20261026"
CANONICAL = ROOT / "docs/architecture/21_SparkX_2_6_4B_Consolidated_Architecture_and_MiniCPM_Validation.md"
CONFIG = ROOT / "configs/sparkx_2_6_4b_dynamic_memory_bridge_study_v1.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checks = []

    def check(name, condition):
        checks.append({"name": name, "passed": bool(condition)})

    config = json.loads(CONFIG.read_text())
    evidence = json.loads((HERE / "evidence.json").read_text())
    snapshot = json.loads((HERE / "repository_snapshot.json").read_text())
    costs = json.loads((HERE / "analytical_costs.json").read_text())
    check("analytical ledger regenerates exactly", costs == calculate(config))
    for flag in ("training_launched", "runtime_implemented", "model_tests_run"):
        check(f"{flag} remains false", config[flag] is False)
    check("added parameter cap removed", config["parameter_policy"]["added_parameter_cap"] is None)
    check("full backbone FT remains required", config["parameter_policy"]["full_offline_language_backbone_finetuning_required"])
    check("LoRA not required for interface", config["parameter_policy"]["lora_required_for_interface"] is False)
    check("proposal is not execution protocol", config["status"] == "proposal_not_execution_protocol")
    for model in ("spark", "minicpm"):
        geometry = config["geometry"][model]
        check(f"{model} exposes all sites", config["ports"][f"exposed_{model}_layers"] == list(range(geometry["layers"])))
        for arm in config["ports"]["active_layer_controls"]:
            layers = arm[f"{model}_read_layers"]
            check(f"{model} {arm['name']} valid distinct indices",
                  len(layers) == len(set(layers)) and all(0 <= i < geometry["layers"] for i in layers))
    memory = config["memory"]
    check("MemT dense control explicit", memory["fast_association"]["memt_mos_is_dense"])
    check("sparse branch requires actual skip", memory["fast_association"]["sparse_actual_skip_required"])
    check("query pass precedes action", memory["record_bank"]["selection_phase"] == "separate_native_known_query_pass_before_action")
    check("one commit independent of reader density", memory["fast_association"]["max_commits_per_observation_boundary"] == 1)
    check("key producer initially stable", memory["fast_addressing"]["initial_key_producer_frozen"])
    check("durable reset includes address and associative state", {"fast_A", "fast_F"} <= set(config["validation"]["clear_for_durable_learning"]))
    check("evaluation updates disabled", config["validation"]["evaluation_updates_disabled"])
    check("32MiB baseline retained", config["resource_profiles"]["old_32MiB_baseline_preserved"])
    check("larger profiles not admitted", config["resource_profiles"]["larger_profiles_not_admitted_runs"])
    check("all parameter/state classes distinct", len(set(config["state_classes"])) == 5)
    check("canonical revision 4.2", "**Revision 4.2" in CANONICAL.read_text())
    original = next(x for x in snapshot["files"] if x["path"] == str(CANONICAL.relative_to(ROOT)))
    check("prior canonical snapshot matches observed bytes", sha(HERE / "architecture_v4_1_snapshot.txt") == original["sha256"])
    for path in (config["baseline_config"], config["density_control_config"]):
        original = next(x for x in snapshot["files"] if x["path"] == path)
        check(f"baseline preserved: {path}", sha(ROOT / path) == original["sha256"])
    check("15 papers with scoped reading", len(evidence["papers"]) == evidence["paper_count"] == 15)
    names = {x["name"] for x in evidence["sources"]}
    for paper in evidence["papers"]:
        check(f"paper {paper['id']} sources and scope", set(paper["sources"]) <= names and bool(paper["reading_scope"]))
    source_checks = 0
    for source in evidence["sources"]:
        if "sha256" in source:
            path = Path(source["scratch_path"])
            check(f"public source hash: {source['name']}", path.is_file() and sha(path) == source["sha256"])
            source_checks += 1
        else:
            check(f"failed source explicitly recorded: {source['name']}", bool(source.get("error")))
    check("retrieval counts consistent", len(evidence["sources"]) == evidence["source_count"] == evidence["source_successes"] + evidence["source_failures"])

    markdown = sorted(DOCS.glob("*.md")) + [HERE / "README.md", CANONICAL]
    local_links = 0
    for path in markdown:
        content = path.read_text()
        check(f"no trailing whitespace: {path.name}", all(line.rstrip() == line for line in content.splitlines()))
        for destination in re.findall(r"\[[^\]]*\]\(([^)]+)\)", content):
            destination = destination.strip("<>").split("#", 1)[0]
            if not destination or re.match(r"[a-zA-Z]+:", destination):
                continue
            target = (path.parent / unquote(destination)).resolve()
            check(f"local link {path.name} -> {destination}", target.exists())
            local_links += 1

    files = markdown + [CONFIG, HERE / "evidence.json", HERE / "repository_snapshot.json",
                        HERE / "analytical_costs.json", HERE / "estimate_costs.py", Path(__file__)]
    failed = [x["name"] for x in checks if not x["passed"]]
    report = {
        "format": "sparkx.dynamic_memory_bridge.documentation_validation.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "documentation_links_arithmetic_config_and_public_source_integrity_only",
        "model_tests_run": False, "training_launched": False,
        "checks": len(checks), "passed": len(checks) - len(failed), "failures": failed,
        "local_links_checked": local_links, "public_source_hashes_checked": source_checks,
        "files": [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in files],
        "details": checks,
    }
    (HERE / "documentation_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("checks", "passed", "failures", "local_links_checked", "public_source_hashes_checked", "model_tests_run")}, indent=2))
    return bool(failed)


if __name__ == "__main__":
    sys.exit(main())

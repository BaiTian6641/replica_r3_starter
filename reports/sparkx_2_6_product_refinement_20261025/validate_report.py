"""Check research artifacts and recorded arithmetic; never import model code."""
import hashlib
import json
import re
from pathlib import Path
from estimate_costs import calculate

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
checks = []


def check(name, condition, detail=None):
    checks.append({"name": name, "pass": bool(condition), "detail": detail})


def read(path):
    return json.loads((ROOT / path).read_text())


plan = read("configs/sparkx_2_6_4b_product_plan_v3.json")
cost = json.loads((OUT / "analytical_costs.json").read_text())
evidence = json.loads((OUT / "evidence.json").read_text())
followup_path = ROOT / plan["repository_followup"]
followup = json.loads(followup_path.read_text())
check("concurrent_followup_revision", followup["head"] == "023e8b9d4dd0dac81ebbea9d50eb65dc7b12d213")
check("concurrent_smoke_not_quality_verdict", followup["findings"]["first_sustained_smoke"]["is_optimizer_quality_verdict"] is False)
check("costs_recompute_exactly", cost == calculate())
check("full_backbone_scope", plan["offline_training"]["original_parameters"] == 4112079360)
check("full_finetuning_required", plan["product"]["full_backbone_finetuning_required"])
check("not_execution_protocol", plan["status"] == "proposal_not_execution_protocol")
check("no_training_claim", plan["training_launched_by_review"] is False)
check("source_revision", plan["models"]["sparkx"]["revision"] == "66e3ae7bee54e0a523d692cb8ef841f35409807b")
check("tensor_inventory_total", cost["spark"]["unique_parameters"] == 4112079360 and cost["spark"]["serialized_tensors"] == 290)
check("adam_fp32_exceeds_guard", cost["spark"]["training_state_GiB"]["fp32_weights_grads_two_fp32_moments_16_bytes"] > 50)
check("mixture_sums_to_one", abs(sum(plan["offline_training"]["training_mix_by_processed_positions"].values()) - 1) < 1e-12)
check("complete_factorial", set(plan["factorial"]["arms"]) == {f"{i:03b}" for i in range(8)})
check("slot_turnover_without_semantic_overflow", 2*256 < 544 <= 768 < 1024)
check("no_silent_streaming_claim", plan["audio"]["candidate"]["causal_checkpoint"] is False)
check("media_budget", cost["proposed_16_second_stream_window"]["total_token_cap"] == plan["long_run"]["total_media_token_cap"] == 640)
check("audio_hour_positions", cost["audio"]["tokens_per_hour_at_10_Hz"] == 36000)
check("hybrid_cache_8k", cost["spark"]["bf16_KV_examples"][0]["hybrid_upper_bound_MiB"] == 342)
check("exclusive_worker", plan["resource_policy"]["exclusive_MPS_worker"] is True)
check("guard_preserved", plan["resource_policy"]["driver_ceiling_bytes"] == 50*2**30)
check("immutable_base_token_ids", plan["modules"]["vocab"]["base_ids_immutable"] is True)
check("no_duplicate_base_tables", plan["modules"]["vocab"]["clone_base_tables"] is False)
check("spark_engineering_independent_of_rsi", plan["spark_engineering_depends_on_rsi"] is False)

source_names = [s["name"] for s in evidence["sources"]]
check("unique_source_ids", len(source_names) == len(set(source_names)))
papers = [s for s in evidence["sources"] if s["source_kind"] == "primary_paper"]
check("selected_paper_count", len(papers) == evidence["selected_full_text_papers"] == 16)
scratch_verified = 0
for source in evidence["sources"]:
    check("source_scope:" + source["name"], bool(source.get("reading_scope")))
    check("source_https:" + source["name"], source["url"].startswith("https://"))
    if "error" in source:
        check("access_failure_excluded_from_content:" + source["name"], source["source_kind"] == "primary_endpoint_access_check")
        continue
    check("source_digest:" + source["name"], bool(re.fullmatch(r"[0-9a-f]{64}", source.get("sha256", ""))))
    scratch = Path(source.get("retrieval_scratch_path", "/not-available"))
    if scratch.is_file():
        check("retrieved_bytes:" + source["name"], hashlib.sha256(scratch.read_bytes()).hexdigest() == source["sha256"])
        scratch_verified += 1

audit = read("reports/vision_curriculum_20261025/finalize_receipts/audit_593dc5406ebea4a09403ef597f81b513ad143025d3823ae0c3e9f45904d505d5.json")
check("sealed_vision_audit", audit["audit"]["score"]["controls"]["correct"]["correct"] == 37 and audit["audit"]["cases"] == 120)
check("vision_rejection_retained", audit["audit"]["learned_quality_pass"] is False)
check("external_caption_failure_retained", audit["external_final"]["caption"]["token_f1"] == 0)
memory = read("runs/memory_update_dynamics_20261025/native01/run_report.json")
check("memory_mechanism", memory["decision"] == "write_position_or_recency_interference")
check("memory_exact_rate", memory["summary"]["shared_exact_top1"] == 0.15)
check("memory_address_rate", abs(memory["summary"]["shared_address_top1"] - 232/240) < 1e-12)
perf = read("reports/m2_ultra_training_performance_20261025/native_summary.json")
check("full_mini_training_scope", perf["trainable_parameters"] == 2516756480 and perf["trainable_tensors"] == 381)
check("short_probe_limit", perf["sequence_length"] == 55 and all(len(a["updates"]) == 3 for a in perf["measured_arms"].values()))

documents = [ROOT / plan["canonical_document"], *sorted((ROOT / "docs/research/sparkx_2_6_product_20261025").glob("*.md")), OUT / "README.md"]
for doc in documents:
    content = doc.read_text()
    check("no_placeholder:" + doc.name, not re.search(r"\b(?:TODO|TBD|INSERT_HERE)\b", content))
    check("balanced_fences:" + doc.name, content.count("```") % 2 == 0)
    for target in re.findall(r"\]\(([^\s)]+)\)", content):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        target = target.split("#", 1)[0]
        path = (doc.parent / target).resolve()
        check("local_link:" + doc.name + ":" + target, path.exists() or path == OUT / "validation.json")

snapshot = json.loads((OUT / "repository_snapshot.json").read_text())
live_changes = []
for entry in snapshot["files"]:
    path = ROOT / entry["path"]
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        live_changes.append(entry["path"])

result = {
    "format": "sparkx.product_refinement.artifact_validation.v1",
    "scope": "documentation, source-byte integrity, raw-report cross-check and arithmetic only; no model or benchmark execution",
    "passed": all(c["pass"] for c in checks), "checks_passed": sum(c["pass"] for c in checks),
    "checks_total": len(checks), "retrieved_source_hashes_verified": scratch_verified,
    "snapshot_files_changed_since_capture": live_changes,
    "snapshot_change_note": "Canonical document is intentionally updated; other changes may be concurrent and do not rewrite the captured historical evidence.",
    "checks": checks,
    "artifact_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in documents + [ROOT / "configs/sparkx_2_6_4b_product_plan_v3.json", OUT / "analytical_costs.json", OUT / "evidence.json", followup_path]},
}
(OUT / "validation.json").write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps({k: result[k] for k in ("passed", "checks_passed", "checks_total", "retrieved_source_hashes_verified", "snapshot_files_changed_since_capture")}))
raise SystemExit(0 if result["passed"] else 1)

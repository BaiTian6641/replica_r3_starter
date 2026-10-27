"""Documentation, public-source and arithmetic checks; never loads a model."""
from pathlib import Path
from urllib.parse import unquote
import hashlib
import json
import re
import runpy

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
checks, unavailable = [], []


def check(name, condition, detail=None):
    checks.append({"name": name, "passed": bool(condition), **({"detail": detail} if detail is not None else {})})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def anchors(text):
    headings = re.findall(r"^#{1,6}\s+(.+)$", text, re.M)
    return {re.sub(r"[^\w\- ]", "", h.lower()).replace(" ", "-") for h in headings}


config_path = ROOT / "configs/sparkx_2_6_4b_final_product_v5.json"
c = json.loads(config_path.read_text())
costs = json.loads((OUT / "analytical_costs.json").read_text())
expected = runpy.run_path(str(OUT / "calculate_costs.py"))["calculate"]()
check("arithmetic_receipt_reproduces", costs == expected)
check("design_not_runtime_admission", not c["execution_authorized_by_this_file"])
check("no_native_execution", not c["native_execution_in_this_review"] and not costs["native_execution"])
check("offline_full_modalities_required", c["invariants"]["offline_full_stage_trains_original_backbone_and_encoders"])
check("deployment_backbone_frozen", c["invariants"]["deployment_original_backbone_frozen"])
check("optional_spine_separate", c["invariants"]["spine_separate_optional"] and not c["spine"]["required_for_intrinsic_memory"])
check("stable_plus_plastic_count", costs["parameters"]["stable_partial"] + costs["parameters"]["plastic"] == costs["parameters"]["combined_partial"])
check("reader_groups_partition_sites", sum(c["intrinsic_integration"]["depth_groups"], []) == c["intrinsic_integration"]["spark_sites"])
check("record_feature_shape", 128+256+2*128+2*128 == c["episodic"]["feature_elements_per_record"])
check("record_packet_bound", c["intrinsic_integration"]["selected_record_union_max"]*c["intrinsic_integration"]["packets_per_selected_record"] == c["intrinsic_integration"]["selected_record_packets_max"])
check("record_KV_bound", costs["derived_payloads"]["selected_record_KV_per_site_KiB"] == 128 and costs["derived_payloads"]["selected_record_KV_all_sites_MiB"] == 1.125)
check("plastic_pool_partition", c["plastic"]["fast_pool"] + c["plastic"]["slow_pool"] == c["plastic"]["experts_stored"])
check("slow_clock_exercised_twice", c["plastic"]["offline_episode_eligible_teaching_events_min"] >= 2*c["plastic"]["slow_every_eligible_events"])
check("two_selected_expert_writes", c["plastic"]["updated_experts_per_transaction_max"] == 2)
check("inactive_expert_no_decay", not c["plastic"]["inactive_expert_step_or_moment_decay"])
check("no_reader_count_write_multiplier", not c["fast_memory"]["one_update_per_reader_site"])
check("fixed_first_order_inner_rate", not c["plastic"]["inner_learning_rate_meta_learned_in_reference"])
check("audio_noncausal_declared", not c["audio"]["causal"] and c["audio"]["availability"] == "window_end_plus_encoding_delay")
check("new_and_original_mlp_distinguished", c["backbone"]["original_mlp_activation"] == "gelu" and c["plastic"]["new_branch_activation"] == "silu")
check("vocab_stable_identity", c["episodic"]["concept_identity_independent_of_vocab_slot"] and not c["vocabulary"]["optimizer_state_keyed_only_by_slot"])
check("factorial_complete_unique", set(c["validation"]["joint_arms"]) == {f"{i:03b}" for i in range(8)} and len(c["validation"]["joint_arms"]) == 8)
check("planning_mixture_sum", abs(sum(c["training"]["planning_processed_position_mixture"].values()) - 1) < 1e-12)
check("KV_reference_342_MiB", costs["KV"]["bounded_sliding_cache_MiB"] == 342)
check("KV_unbounded_control_1_125_GiB", costs["KV"]["all_layers_full_cache_GiB"] == 1.125)
check("AV_reference_640_positions", costs["derived_payloads"]["AV_example_positions"] == 640)
check("phrase_buffer_1_25_MiB", costs["derived_payloads"]["phrase_rows_MiB"] == 1.25)
check("128_MiB_profile_not_fit", not costs["dynamic"]["128_MiB_live_profile_fits"])
check("unchanged_50_GiB_guard", c["resource_profile"]["full_training_guard_unchanged_GiB"] == 50)

evidence = json.loads((OUT / "evidence.json").read_text())
check("19_fresh_10_retained_sources", sum(r["retrieval_in_this_review"] for r in evidence["records"]) == 19 and len(evidence["records"]) == 29)
for r in evidence["records"]:
    path = Path(r["scratch_path"])
    if path.exists():
        check("source_hash:" + r["name"], sha(path) == r["sha256"])
    else:
        unavailable.append({"source": r["name"], "url": r["url"], "expected_sha256": r["sha256"]})
by_name = {r["name"]: r for r in evidence["records"]}
spark_cfg_path = Path(by_name["final_spark_config"]["scratch_path"])
if spark_cfg_path.exists():
    sc = json.loads(spark_cfg_path.read_text())
    for target, origin in [("layers", "num_hidden_layers"), ("width", "hidden_size"), ("ffn_width", "intermediate_size"), ("query_heads", "num_attention_heads"), ("kv_heads", "num_key_value_heads"), ("head_width", "head_dim"), ("base_vocab", "vocab_size"), ("original_mlp_activation", "hidden_act"), ("sliding_window", "sliding_window"), ("embedding_head_tied", "tie_word_embeddings")]:
        check("pinned_spark:"+target, c["backbone"][target] == sc[origin])
    check("pinned_full_attention_sites", [i for i,x in enumerate(sc["layer_types"]) if x == "full_attention"] == c["backbone"]["full_attention_layers"])
    check("pinned_QKV_geometry", (sc["num_attention_heads"]+2*sc["num_key_value_heads"])*sc["head_dim"] == c["backbone"]["qkv_width"])
    check("pinned_attention_output_geometry", sc["num_attention_heads"]*sc["head_dim"] == c["backbone"]["attention_output_width"])
code_path = Path(by_name["final_spark_code"]["scratch_path"])
if code_path.exists():
    code = code_path.read_text()
    check("pinned_code_GELU_guard", 'if config.hidden_act != "gelu"' in code)
    check("pinned_code_tied_head", "F.linear(hidden_states, embed_weight)" in code)
    local = ROOT / "reports/spark_preparation_20261025/run_01/assets/official_bf16_control/modeling_spark.py"
    check("fresh_code_matches_repository_pin", sha(code_path) == sha(local))
spear_cfg_path = Path(by_name["final_spear_config"]["scratch_path"])
if spear_cfg_path.exists():
    check("pinned_SPEAR_noncausal", json.loads(spear_cfg_path.read_text())["causal"] is False)

docs = [ROOT / c["document"], OUT / "README.md"]
for doc in docs:
    body = doc.read_text()
    check("balanced_code_fences:"+doc.name, sum(line.startswith("```") for line in body.splitlines()) % 2 == 0)
    for index, target in enumerate(re.findall(r"\[[^\]\n]+\]\(([^)]+)\)", body)):
        if target.startswith(("https://", "http://", "mailto:")):
            continue
        path_part, _, fragment = target.partition("#")
        path = (doc.parent / unquote(path_part)).resolve() if path_part else doc
        exists = path.exists() or path == OUT / "verification.json"
        check(f"local_link:{doc.name}:{index}", exists, target if not exists else None)
        if exists and fragment and path.suffix == ".md":
            check(f"anchor:{doc.name}:{index}", fragment in anchors(path.read_text()), target)

main_text = docs[0].read_text()
check("26_product_sections", len(re.findall(r"^## \d+\.", main_text, re.M)) == 26)
check("open_issues_explicit", all(f"| O{i} |" in main_text for i in range(1,17)))
trace = json.loads((OUT / "requirements_traceability.json").read_text())
check("20_requirements_unique", len(trace["requirements"]) == len({r["id"] for r in trace["requirements"]}) == 20)
for r in trace["requirements"]:
    check("requirement_sections:"+r["id"], all(1 <= int(s) <= 26 for s in r["sections"].split(",")))

snap = json.loads((OUT / "repository_snapshot.json").read_text())
preserved = {
    "docs/architecture/21_SparkX_2_6_4B_Consolidated_Architecture_and_MiniCPM_Validation.md": "architecture_v4_5_before.txt",
    "docs/architecture/22_State_Mechanism_and_RSI_Contracts.md": "state_contract_before.txt",
    "docs/architecture/README.md": "architecture_index_before.txt",
}
drift = []
for entry in snap["input_files"]:
    path = ROOT / entry["path"]
    if entry["path"] in preserved:
        check("preserved_input:"+entry["path"], sha(OUT/preserved[entry["path"]]) == entry["sha256"])
    if sha(path) != entry["sha256"]:
        drift.append({"path": entry["path"], "reviewed_sha256": entry["sha256"], "current_sha256": sha(path), "planned_index_edit": entry["path"] in ["docs/architecture/README.md", "docs/architecture/21_SparkX_2_6_4B_Consolidated_Architecture_and_MiniCPM_Validation.md"]})
    elif entry["path"] not in preserved:
        check("unchanged_input:"+entry["path"], True)

acceptance = json.loads((ROOT / "reports/memory_acceptance_20261026/causal_fresh_restart02_acceptance_v3.json").read_text())
quality = acceptance["metrics"]["variants"][acceptance["candidate_variant"]]["quality"]
check("reviewed_memory_verdict_rejected", acceptance["verdict"]["decision"] == "rejected")
check("reviewed_memory_denominators", (quality["accepted_correct"], quality["fact_n"], quality["accepted"]) == (21,384,138))

result = {
    "kind": "documentation_sources_and_arithmetic_verification_not_model_qualification",
    "status": "failed" if any(not x["passed"] for x in checks) else "passed" if not unavailable else "passed_local_checks_sources_partly_unavailable",
    "checks_count": len(checks), "passed": sum(x["passed"] for x in checks),
    "failed": [x for x in checks if not x["passed"]],
    "unavailable_source_cache": unavailable,
    "input_changes_since_snapshot": drift,
    "blueprint_words": len(main_text.split()),
    "native_execution": False, "training": False,
    "checks": checks,
}
(OUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k:result[k] for k in ["status", "checks_count", "passed", "failed", "unavailable_source_cache", "blueprint_words"]}))
if drift:
    print(json.dumps({"input_changes_since_snapshot": drift}))
if result["failed"]:
    raise SystemExit(1)

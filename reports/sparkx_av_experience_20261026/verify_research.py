#!/usr/bin/env python3
"""Check this research package using only standard-library, non-runtime operations."""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent
DOCS = ROOT / "docs/research/sparkx_av_experience_20261026"
CANONICAL = ROOT / "docs/architecture/21_SparkX_2_6_4B_Consolidated_Architecture_and_MiniCPM_Validation.md"
CONFIG = ROOT / "configs/sparkx_2_6_4b_product_plan_v4.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checks = []

    def check(label, condition):
        checks.append({"check": label, "passed": bool(condition)})

    plan = json.loads(CONFIG.read_text())
    costs = json.loads((REPORT / "analytical_costs.json").read_text())
    evidence = json.loads((REPORT / "evidence.json").read_text())
    observation = json.loads((REPORT / "repository_snapshot.json").read_text())
    check("revision_4_schema", plan["format"].endswith(".v4"))
    check("proposal_only", plan["status"] == "proposal_not_execution_protocol")
    check("no_training_started", plan["training_launched_by_review"] is False)
    check("no_runtime_implemented", plan["runtime_implemented_by_review"] is False)
    check("no_downloaded_code_executed", evidence["downloaded_remote_code_executed"] is False)
    check("existing_execution_protocols_preserved", plan["preserve_existing_execution_protocols"] is True)
    check("full_original_backbone_scope", plan["offline_training"]["original_parameters"] == 4112079360)
    check("native_semantic_decisions", plan["product"]["native_semantic_decisions"] is True)
    check("canonical_revision_4", "Revision 4 · 26 October 2026" in CANONICAL.read_text())
    check("snapshot_hash", digest(REPORT / "repository_snapshot.json") == evidence["local_observation"]["sha256"])
    check("v3_snapshot_hash", digest(REPORT / "architecture_v3_snapshot.txt") == evidence["prior_canonical_snapshot"]["sha256"])
    old_canonical = next(x for x in observation["files"] if x["path"] == str(CANONICAL.relative_to(ROOT)))
    check("v3_snapshot_preserves_observed_canonical", digest(REPORT / "architecture_v3_snapshot.txt") == old_canonical["sha256"])

    b = plan["modules"]["bank"]
    core = plan["modules"]["core"]
    bridge = plan["modules"]["native_bridge"]
    fast = plan["modules"]["fast_memory"]
    vocab = plan["modules"]["vocab"]
    spark = plan["models"]["sparkx"]
    width = spark["hidden_size"]
    mib = 2 ** 20
    elements = b["key_width"] + b["value_width"] + b["visual_views"] * b["visual_view_width"] + b["audio_views"] * b["audio_view_width"]
    check("bank_width", elements == b["elements_per_record"] == costs["bank_elements_per_record"] == 896)
    check("bank_bytes_per_record", costs["bank_bytes_per_record_bf16"] == elements * 2)
    check("spark_bank_payload", costs["spark_tensor_bank_MiB"] == b["spark_entries"] * elements * 2 / mib == 7)
    check("mini_bank_payload", costs["minicpm_tensor_bank_MiB"] == b["minicpm_entries"] * elements * 2 / mib == 1.75)
    check("v3_control_payload", costs["v3_control_spark_tensor_bank_MiB"] == b["spark_entries"] * (b["key_width"] + b["value_width"]) * 2 / mib == 3)
    check("metadata_allowance", costs["spark_fixed_metadata_cap_MiB"] == b["spark_entries"] * b["fixed_metadata_bytes_per_record_cap"] / mib == 2)
    fast_bytes = fast["matrix_count"] * fast["matrix_shape"][0] * fast["matrix_shape"][1] * 4
    check("fast_state_payload", costs["optional_fast_state_MiB"] == fast_bytes / mib == 0.25)
    check("fast_state_shared", fast["shared_across_ports"] is True and fast["default_enabled"] is False)
    projections = len(spark["ports_zero_indexed"]) * (width * b["key_width"] + (b["value_width"] + core["width"]) * width)
    check("projection_weights", costs["spark_port_query_output_weights_lower_bound"] == projections == 3276800)
    check("projection_bytes", costs["spark_port_query_output_weights_bf16_MiB"] == projections * 2 / mib == 6.25)
    scan = b["spark_entries"] * b["key_width"]
    check("one_scan_MACs", costs["single_spark_bank_scan_MACs"] == scan)
    check("two_scan_MACs", costs["two_spark_bank_scan_MACs"] == scan * b["full_bank_scans_per_query_boundary_cap"])
    check("shared_event_slots", sum(bridge["initial_slot_partition"].values()) == bridge["event_slots_shared_cap"] == plan["vision"]["event_slots_cap"] == 64)
    check("query_independent_writes", b["unknown_future_query_writing"] is True and b["query_known_during_write"] is False)
    check("read_union_bound", b["per_port_topk"] * len(spark["ports_zero_indexed"]) == b["selected_unique_records_cap"] == 16)
    check("stable_producer", b["stable_experience_producer_within_online_release"] is True)
    check("bounded_auxiliary_state", b["prototype_and_rule_records_share_capacity"] is True and b["auxiliary_history_bounded"] is True)
    known_bytes = b["spark_entries"] * (elements * 2 + b["fixed_metadata_bytes_per_record_cap"]) + bridge["event_slots_shared_cap"] * bridge["width"] * 4 + core["slots"] * core["width"] * 4 + fast_bytes + vocab["active_slots"] * width * 2
    check("known_state_payload", costs["known_resident_state_payload_MiB_including_optional_fast_and_active_phrase_rows"] == known_bytes / mib)
    cap = plan["long_run"]["resident_state_MiB_cap"]
    check("remaining_logical_quota", costs["logical_resident_cap_remaining_after_known_payload_MiB"] == cap - known_bytes / mib > 0)
    run = plan["long_run"]
    positions = run["frames_per_window"] * run["tokens_per_frame"] + run["additional_crops_max"] * run["tokens_per_crop"] + run["audio_tokens"] + run["event_tokens"] + run["clock_tokens_reserve"]
    check("media_budget", positions == run["total_media_token_cap"] == costs["media_positions"] == 640)
    check("not_a_runtime_estimate", costs["runtime_feasibility_established"] is False)
    check("factorial_kept", len(plan["factorial"]["arms"]) == 8 and plan["factorial"]["bit_order"] == ["M", "V", "L"])
    check("scope_of_V", plan["validation_extensions"]["factorial_V_means"] == "swappable_vocabulary_not_vision")
    check("retention_margin", plan["validation_extensions"]["retention_margin_pp"] == plan["release_gates"]["retention_noninferiority_margin_pp"] == 2)
    check("five_point_new_gates", plan["validation_extensions"]["primary_additional_gain_min_pp"] == 5)
    reset = plan["validation_extensions"]["durable_parameter_transfer_reset"]
    check("parameter_transfer_resets_transient_state", {"fast_weights", "core_and_event_state", "record_read_cache", "teaching_derived_lexical_candidates", "source_KV", "derived_prototypes_and_rules"}.issubset(reset["remove_or_reset"]))
    check("parameter_transfer_eval_updates_disabled", reset["updates_during_evaluation"] is False and reset["test_prompt_restates_taught_rule"] is False)
    check("online_modality_position_accounting", plan["modules"]["learning"]["online_position_cap_includes_inserted_modality_positions"] is True)
    check("exclusive_worker", plan["resource_policy"]["exclusive_MPS_worker"] is True)
    check("driver_guard", plan["resource_policy"]["driver_ceiling_bytes"] == 50 * 2 ** 30)
    check("cpu_fallback_forbidden", plan["resource_policy"]["silent_CPU_fallback"] is False)
    memory = observation["findings"]["memory"]
    check("memory_rejection_preserved", memory["decision"] == plan["evidence_update"]["causal_refresh"]["decision"] == "rejected")
    check("causal_score", abs(memory["scores"]["shared_mean/causal_query_refresh"]["exact_top1"] - 110 / 240) < 1e-12)
    check("zero_optimizer_updates", memory["optimizer_updates"] == 0)
    check("restart_failure_preserved", memory["restart_failure"] == "Fresh-process outputs differ from the first process")
    check("fourteen_papers", evidence["paper_count"] == len(evidence["papers"]) == 14)

    source_names = {s["name"] for s in evidence["sources"]}
    for paper in evidence["papers"]:
        check("paper_source_references:" + paper["id"], bool(paper["reading_scope"]) and set(paper["source_names"]).issubset(source_names))
    available_sources, missing_scratch = [], []
    for source in evidence["sources"]:
        if "sha256" not in source:
            check("failed_source_not_evidence:" + source["name"], source.get("claim_eligible") is False)
            continue
        path = Path(source["scratch_path"])
        if path.is_file():
            check("source_hash:" + source["name"], digest(path) == source["sha256"])
            available_sources.append(source["name"])
        else:
            missing_scratch.append(source["name"])
        text_path = Path(source["text_path"])
        if text_path.is_file():
            check("extracted_text_hash:" + source["name"], digest(text_path) == source["extracted_text_sha256"])
    mirrors = [s["sha256"] for s in evidence["sources"] if s["name"] in {"tvp_mirror_a", "tvp_mirror_b"}]
    check("mirror_hash_equality", len(mirrors) == 2 and len(set(mirrors)) == 1 and mirrors[0] == evidence["deepseek_copies"]["sha256"])
    check("mirror_authentication_not_inferred", evidence["deepseek_copies"]["author_authentication_established"] is False)

    documents = [CANONICAL, *sorted(DOCS.glob("*.md")), REPORT / "README.md"]
    local_links = 0
    for document in documents:
        text = document.read_text()
        check("balanced_code_fences:" + str(document.relative_to(ROOT)), sum(line.startswith("```") for line in text.splitlines()) % 2 == 0)
        for raw in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
            target = raw.strip().strip("<>")
            parts = urlsplit(target)
            if parts.scheme or parts.netloc or not parts.path:
                continue
            path = (document.parent / unquote(parts.path)).resolve()
            # Output is written below after all checks, so its link is intentionally forward-declared.
            exists = path.exists() or path == REPORT / "documentation_validation.json"
            check("local_link:" + str(document.relative_to(ROOT)) + "->" + target, exists)
            local_links += 1

    concurrent_status = []
    for entry in observation["files"]:
        path = ROOT / entry["path"]
        concurrent_status.append({"path": entry["path"], "exists_now": path.exists(), "matches_observation_hash_now": digest(path) == entry["sha256"] if path.is_file() else None, "canonical_intentionally_revised": path == CANONICAL})
    artifacts = [*documents, CONFIG, REPORT / "analytical_costs.json", REPORT / "evidence.json", REPORT / "repository_snapshot.json", REPORT / "architecture_v3_snapshot.txt", Path(__file__).resolve()]
    failures = [c for c in checks if not c["passed"]]
    result = {
        "format": "sparkx.av_experience.documentation_validation.v1",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "documentation_configuration_evidence_hash_and_arithmetic_only",
        "model_or_device_tests_run": False,
        "network_used_by_validator": False,
        "passed": not failures,
        "checks_count": len(checks),
        "failures": failures,
        "local_link_count": local_links,
        "scratch_sources_hash_checked": available_sources,
        "scratch_sources_unavailable": missing_scratch,
        "observed_repository_files_current_status": concurrent_status,
        "artifact_hashes": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in artifacts],
        "checks": checks,
    }
    (REPORT / "documentation_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["scope", "passed", "checks_count", "local_link_count", "failures", "scratch_sources_unavailable"]}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

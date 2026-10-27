"""Documentation, tracked-evidence and arithmetic checks; never loads a model.

The verifier is self-contained from tracked repository evidence.  It accepts an
explicit ``--root`` (the repository) and ``--out`` (the report directory holding
the portable evidence pins and the versioned receipt).  Temporary source-cache
files are optional cross-checks only; a clean checkout without them still runs
every required check.

Output: ``verification_v2.json`` in ``--out``.  The historical
``verification.json`` receipt is preserved and never rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
OUTPUT_NAME = "verification_v2.json"
FORMAT = "replica.spark_final_product_verification.v2"
HISTORICAL_CHECK_COUNT = 223


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_if(path: Path):
    return _load(path) if path.is_file() else None


def _tracked(root: Path, relative: str, cache: dict[str, bool] | None = None) -> bool:
    if cache is not None and relative in cache:
        return cache[relative]
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", relative],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        tracked = result.returncode == 0
    except OSError:
        tracked = False
    if cache is not None:
        cache[relative] = tracked
    return tracked


def _matches_index(root: Path, relative: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "diff", "--quiet", "--", relative],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0
    except OSError:
        return False


def _canonical_digest(entries: list[dict]) -> str:
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _anchors(text: str) -> set[str]:
    headings = re.findall(r"^#{1,6}\s+(.+)$", text, re.M)
    return {re.sub(r"[^\w\- ]", "", h.lower()).replace(" ", "-") for h in headings}


def _close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return abs(left - right) <= tolerance


def _calculator(root: Path):
    path = root / "reports/sparkx_final_product_20261026/calculate_costs.py"
    spec = importlib.util.spec_from_file_location("sparkx_calculate_costs", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load calculator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = Path(root)
    return module.calculate()


def verify(root: Path = ROOT, out: Path = OUT, *, write: bool = True) -> dict:
    root = Path(root).resolve()
    out = Path(out).resolve()
    checks: list[dict] = []
    unavailable: list[dict] = []
    reviewed_applied: list[dict] = []
    unexplained: list[dict] = []
    historical_manifest_report: dict = {}
    final_manifest_report: dict = {}

    def check(name: str, condition, detail=None) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(condition),
                **({"detail": detail} if detail is not None else {}),
            }
        )

    portable = _load(out / "portable_evidence.json")
    check(
        "portable_paths_are_relative",
        all(
            not record["path"].startswith("/") and ".." not in Path(record["path"]).parts
            for group in ("required_receipts", "required_configs")
            for record in portable.get(group, [])
        ),
    )

    # ------------------------------------------------------------------
    # Declarative arithmetic
    # ------------------------------------------------------------------
    config_path = root / "configs/sparkx_2_6_4b_final_product_v5.json"
    c = _load(config_path)
    costs = _load(out / "analytical_costs.json")
    expected = _calculator(root)
    check("arithmetic_receipt_reproduces", costs == expected)
    check("design_not_runtime_admission", not c["execution_authorized_by_this_file"])
    check("no_native_execution", not c["native_execution_in_this_review"] and not costs["native_execution"])
    check("offline_full_modalities_required", c["invariants"]["offline_full_stage_trains_original_backbone_and_encoders"])
    check("deployment_backbone_frozen", c["invariants"]["deployment_original_backbone_frozen"])
    check("optional_spine_separate", c["invariants"]["spine_separate_optional"] and not c["spine"]["required_for_intrinsic_memory"])
    check("stable_plus_plastic_count", costs["parameters"]["stable_partial"] + costs["parameters"]["plastic"] == costs["parameters"]["combined_partial"])
    check("reader_groups_partition_sites", sum(c["intrinsic_integration"]["depth_groups"], []) == c["intrinsic_integration"]["spark_sites"])
    check("record_feature_shape", 128 + 256 + 2 * 128 + 2 * 128 == c["episodic"]["feature_elements_per_record"])
    check("record_packet_bound", c["intrinsic_integration"]["selected_record_union_max"] * c["intrinsic_integration"]["packets_per_selected_record"] == c["intrinsic_integration"]["selected_record_packets_max"])
    check("record_KV_bound", costs["derived_payloads"]["selected_record_KV_per_site_KiB"] == 128 and costs["derived_payloads"]["selected_record_KV_all_sites_MiB"] == 1.125)
    check("plastic_pool_partition", c["plastic"]["fast_pool"] + c["plastic"]["slow_pool"] == c["plastic"]["experts_stored"])
    check("slow_clock_exercised_twice", c["plastic"]["offline_episode_eligible_teaching_events_min"] >= 2 * c["plastic"]["slow_every_eligible_events"])
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

    # ------------------------------------------------------------------
    # Public-source bindings from tracked pins
    # ------------------------------------------------------------------
    evidence = _load(out / "evidence.json")
    check("19_fresh_10_retained_sources", sum(r["retrieval_in_this_review"] for r in evidence["records"]) == 19 and len(evidence["records"]) == 29)
    by_name = {r["name"]: r for r in evidence["records"]}
    source_cache_cross_checks: list[dict] = []
    for record in evidence["records"]:
        path = Path(record["scratch_path"])
        if path.exists():
            source_cache_cross_checks.append(
                {
                    "source": record["name"],
                    "sha256_matches": _sha256(path) == record["sha256"],
                    "required": False,
                }
            )
        else:
            unavailable.append({"source": record["name"], "url": record["url"], "expected_sha256": record["sha256"]})

    config_pin = by_name.get("final_spark_config")
    code_pin = by_name.get("final_spark_code")
    spear_pin = by_name.get("final_spear_config")
    for pin in portable.get("source_pins", []):
        record = by_name.get(pin.get("evidence_record"))
        check(
            "source_pin_binding:" + pin["name"],
            record is not None and record["sha256"] == pin["sha256"] and bool(record["url"]),
        )
        tracked_path = pin.get("tracked_path")
        if tracked_path:
            path = root / tracked_path
            check("source_pin_present:" + pin["name"], path.is_file())
            if not path.is_file():
                continue
            check("source_pin_hash:" + pin["name"], _sha256(path) == pin["sha256"])
            if pin["name"] == "final_spark_config":
                sc = _load(path)
                for target, origin in pin["extract"].items():
                    check("pinned_spark:" + target, c["backbone"][target] == sc[origin])
                check(
                    "pinned_full_attention_sites",
                    [i for i, kind in enumerate(sc["layer_types"]) if kind == "full_attention"] == c["backbone"]["full_attention_layers"],
                )
                check("pinned_QKV_geometry", (sc["num_attention_heads"] + 2 * sc["num_key_value_heads"]) * sc["head_dim"] == c["backbone"]["qkv_width"])
                check("pinned_attention_output_geometry", sc["num_attention_heads"] * sc["head_dim"] == c["backbone"]["attention_output_width"])
            elif pin["name"] == "final_spark_code":
                code = path.read_text(encoding="utf-8")
                for guard in pin["reviewed_guards"]:
                    check("pinned_code_guard:" + guard[:24], guard in code)
        else:
            check(
                "source_pin_reviewed_fields:" + pin["name"],
                bool(pin.get("reviewed_fields")) and bool(pin.get("provenance_note")),
            )
    if config_pin is not None:
        check("pinned_config_matches_repository_pin", _sha256(root / "reports/spark_preparation_20261025/run_01/assets/official_bf16_control/config.json") == config_pin["sha256"])
    if code_pin is not None:
        check("fresh_code_matches_repository_pin", _sha256(root / "reports/spark_preparation_20261025/run_01/assets/official_bf16_control/modeling_spark.py") == code_pin["sha256"])
    if spear_pin is not None:
        reviewed = {pin["name"]: pin for pin in portable.get("source_pins", [])}.get("final_spear_config", {})
        check("pinned_SPEAR_noncausal", reviewed.get("reviewed_fields", {}).get("causal") is False and spear_pin["sha256"] == reviewed.get("sha256"))

    # ------------------------------------------------------------------
    # Documentation links and anchors
    # ------------------------------------------------------------------
    docs = [root / c["document"], out / "README.md"]
    tracked_cache: dict[str, bool] = {}
    generated_receipt = root / "reports/sparkx_final_product_20261026" / OUTPUT_NAME
    for doc in docs:
        body = doc.read_text(encoding="utf-8")
        check("balanced_code_fences:" + doc.name, sum(line.startswith("```") for line in body.splitlines()) % 2 == 0)
        for index, target in enumerate(re.findall(r"\[[^\]\n]+\]\(([^)]+)\)", body)):
            if target.startswith(("https://", "http://", "mailto:")):
                continue
            path_part, _, fragment = target.partition("#")
            path = (doc.parent / unquote(path_part)).resolve() if path_part else doc
            if path == generated_receipt:
                check(f"local_link_generated:{doc.name}:{index}", True)
            else:
                exists = path.exists()
                check(f"local_link:{doc.name}:{index}", exists, target if not exists else None)
                if exists and path.is_relative_to(root):
                    relative = path.relative_to(root).as_posix()
                    check(
                        f"local_link_tracked:{doc.name}:{index}",
                        _tracked(root, relative, tracked_cache),
                        target,
                    )
            if path.exists() and fragment and path.suffix == ".md":
                check(f"anchor:{doc.name}:{index}", fragment in _anchors(path.read_text(encoding="utf-8")), target)

    main_text = docs[0].read_text(encoding="utf-8")
    check("26_product_sections", len(re.findall(r"^## \d+\.", main_text, re.M)) == 26)
    check("open_issues_explicit", all(f"| O{i} |" in main_text for i in range(1, 17)))
    trace = _load(out / "requirements_traceability.json")
    check("20_requirements_unique", len(trace["requirements"]) == len({r["id"] for r in trace["requirements"]}) == 20)
    for requirement in trace["requirements"]:
        check("requirement_sections:" + requirement["id"], all(1 <= int(s) <= 26 for s in requirement["sections"].split(",")))

    # ------------------------------------------------------------------
    # Snapshot drift: reviewed changes are separate records; the rest fail
    # ------------------------------------------------------------------
    snapshot = _load(out / "repository_snapshot.json")
    reviewed = _load(out / "reviewed_input_changes.json")
    reviewed_by_path = {change["path"]: change for change in reviewed.get("snapshot_changes", [])}
    preserved = {
        "docs/architecture/21_SparkX_2_6_4B_Consolidated_Architecture_and_MiniCPM_Validation.md": "architecture_v4_5_before.txt",
        "docs/architecture/22_State_Mechanism_and_RSI_Contracts.md": "state_contract_before.txt",
        "docs/architecture/README.md": "architecture_index_before.txt",
    }
    for entry in snapshot["input_files"]:
        relative = entry["path"]
        path = root / relative
        if relative in preserved:
            check("preserved_input:" + relative, _sha256(out / preserved[relative]) == entry["sha256"])
        if not path.is_file():
            check("snapshot_input_present:" + relative, False)
            unexplained.append({"path": relative, "reason": "missing_snapshot_input"})
            continue
        actual = _sha256(path)
        if actual == entry["sha256"]:
            if relative not in preserved:
                check("unchanged_input:" + relative, True)
            continue
        change = reviewed_by_path.get(relative)
        if (
            change
            and change.get("prior_sha256") == entry["sha256"]
            and change.get("current_sha256") == actual
        ):
            reviewed_applied.append(
                {"path": relative, "kind": change.get("kind"), "reason": change.get("reason")}
            )
            check("reviewed_input_change:" + relative, True)
        else:
            unexplained.append(
                {
                    "path": relative,
                    "reviewed_sha256": entry["sha256"],
                    "current_sha256": actual,
                    "reviewed_record": change,
                }
            )
            check("unexplained_input_drift:" + relative, False, actual)
    check("no_unexplained_snapshot_drift", not unexplained, unexplained or None)

    # ------------------------------------------------------------------
    # Tracked receipt and configuration pins
    # ------------------------------------------------------------------
    for record in portable.get("required_receipts", []):
        path = root / record["path"]
        check("receipt_present:" + record["role"], path.is_file())
        if path.is_file():
            check("receipt_hash:" + record["role"], _sha256(path) == record["sha256"])
            if record.get("format") and path.suffix == ".json":
                check("receipt_format:" + record["role"], _load(path).get("format") == record["format"])
    for record in portable.get("required_configs", []):
        path = root / record["path"]
        check("config_present:" + record["path"], path.is_file())
        if path.is_file():
            check("config_hash:" + record["path"], _sha256(path) == record["sha256"])

    # ------------------------------------------------------------------
    # Memory acceptance v4 is current; v3 is historical
    # ------------------------------------------------------------------
    acceptance_path = root / "reports/memory_acceptance_20261026/causal_fresh_restart02_acceptance_v4.json"
    acceptance = _load_if(acceptance_path)
    check("current_acceptance_v4_present", acceptance is not None)
    if acceptance is not None:
        quality = acceptance["metrics"]["variants"][acceptance["candidate_variant"]]["quality"]
        check("current_acceptance_v4_format", acceptance["format"] == "replica.memory_acceptance.metrics.v4")
        check("current_acceptance_v4_rejected", acceptance["verdict"]["decision"] == "rejected")
        check(
            "current_acceptance_v4_denominators",
            (quality["accepted_correct"], quality["fact_n"], quality["accepted"]) == (21, 384, 138),
        )
        check(
            "current_acceptance_v4_precision",
            _close(quality["accepted_precision"], 21 / 138),
        )
        check("current_acceptance_v4_coverage_valid", acceptance["verdict"]["coverage_valid"] is True and acceptance["verdict"]["fail_closed_on_missing_coverage"] is True)
    v4_manifest = _load_if(root / "reports/memory_acceptance_20261026/receipt_manifest_v4.restart02.json")
    check("v4_manifest_present", v4_manifest is not None)
    if v4_manifest is not None:
        authoritative = v4_manifest["authoritative"]
        check(
            "v4_manifest_selects_current_receipt",
            authoritative["status"] == "current"
            and Path(authoritative["path"]).name == acceptance_path.name
            and acceptance is not None
            and authoritative["file_sha256"] == _sha256(acceptance_path),
        )
        v3_records = [record for record in v4_manifest["receipts"] if record["path"] == "causal_fresh_restart02_acceptance_v3.json"]
        check("v4_manifest_retains_v3_as_historical", bool(v3_records) and v3_records[0].get("not_v4") is True and v3_records[0]["status"] == "historical")
    v3 = _load_if(root / "reports/memory_acceptance_20261026/causal_fresh_restart02_acceptance_v3.json")
    check("historical_acceptance_v3_present", v3 is not None)
    if v3 is not None:
        check("historical_acceptance_v3_readable", v3["format"] == "replica.memory_acceptance.metrics.v3" and v3["verdict"]["decision"] == "rejected")
        check("v3_is_not_v4", v3.get("metric_version") != "memory.acceptance.v4")

    # ------------------------------------------------------------------
    # Calibration rejection
    # ------------------------------------------------------------------
    calibration = _load_if(root / "reports/memory_acceptance_20261026/calibration_dev01.json")
    check("calibration_present", calibration is not None)
    if calibration is not None:
        classification = calibration["gate_classification"]["disposition"]
        check("calibration_format", calibration["format"] == "replica.memory_acceptance.calibration.receipt.v1")
        check("calibration_rejected", classification["selected_rule"] == "oracle_cannot_meet_gates" and classification["stop"] is True)
        check(
            "calibration_no_promotion_or_native_execution",
            classification["candidate_meets_value_gates"] is False
            and classification["heldout_confirmation_permitted"] is False
            and classification["native_execution_permitted"] is False
            and classification["writer_training_permitted"] is False,
        )
        check(
            "calibration_heldout_sealed",
            calibration["audit"]["heldout_rows_scored"] == 0 and calibration["claims"]["heldout_metrics_not_reported"] is True,
        )
        frontier = calibration["frontier_summary"]
        check(
            "calibration_frontier_no_pass",
            frontier["value_points_passing_all_gates"] == 0
            and frontier["value_point_indices_passing_all_gates"] == []
            and frontier["address_points_meeting_hit_and_precision"] == 0,
        )

    # ------------------------------------------------------------------
    # Independent plastic continuation: exactness and 16-expert scope
    # ------------------------------------------------------------------
    comparison_path = root / "reports/nested_text_continuation_comparison_20261026/comparison.json"
    comparison = _load_if(comparison_path)
    protocol_path = root / "configs/nested_text_continuation_comparison_v1.json"
    protocol = _load_if(protocol_path)
    check("continuation_comparison_present", comparison is not None)
    check("continuation_protocol_present", protocol is not None)
    if comparison is not None and protocol is not None:
        check(
            "continuation_exact_match",
            comparison["status"] == "exact_match"
            and comparison["exact_status"] == "passed"
            and comparison["exact_only"] is True
            and comparison["failure_count"] == 0
            and comparison["earliest_divergence"] is None
            and comparison["variation_status"] == "none",
        )
        check(
            "continuation_no_promotion_or_capability_claim",
            comparison["claims"]["promotion"] is False
            and comparison["claims"]["heldout_opened"] is False
            and comparison["claims"]["numeric_envelope"] is False
            and comparison["numeric_envelope"] is False
            and protocol["promotion"] is False,
        )
        check("continuation_16_expert_scope", protocol["bank"]["experts"] == 16 and protocol["one_clock_control_only"] is True)
        check(
            "continuation_event_scope",
            comparison["events"]["continuous_events"] == 28
            and comparison["events"]["parent_events"] == 27
            and comparison["events"]["aa_states_compared"] == 57
            and comparison["events"]["boundary_event"] == "correction-0003",
        )
        check(
            "continuation_protocol_binding",
            comparison["protocol"]["config_sha256"] == _sha256(protocol_path) == protocol.get("protocol_sha256", comparison["protocol"]["config_sha256"]),
        )
        boundary = comparison["comparisons"]["boundary"]
        before_hashes = {
            boundary[key] for key in boundary if key.endswith("_before_sha256")
        }
        after_hashes = {
            boundary[key] for key in boundary if key.endswith("_after_sha256")
        }
        check(
            "continuation_boundary_exact",
            before_hashes == {"31d6821fa845b64b1f3af4ffe0d09627eb256bf20902813c64a26d42d5098574"}
            and after_hashes == {"59cd0c520f869fbadcbca0061a7d86bfa76235063618ea3f41e1e6e50cbf4bf9"},
        )
        final_hashes = {
            comparison["artifacts"]["continuous-a"]["canonical_state_sha256"],
            comparison["artifacts"]["continuous-b"]["canonical_state_sha256"],
            comparison["artifacts"]["resume"]["after_canonical_sha256"],
            comparison["comparisons"]["boundary"]["continuous_a_after_sha256"],
            comparison["comparisons"]["boundary"]["continuous_b_after_sha256"],
            comparison["comparisons"]["boundary"]["resume_after_sha256"],
        }
        check(
            "continuation_final_state_exact",
            final_hashes == {"59cd0c520f869fbadcbca0061a7d86bfa76235063618ea3f41e1e6e50cbf4bf9"},
        )
        for name, guard in comparison["guards"].items():
            guard_path = Path(guard["path"])
            expected_path = root / "reports/nested_text_continuation_comparison_20261026" / guard_path.name
            check("continuation_guard_present:" + name, expected_path.is_file())
            if expected_path.is_file():
                check("continuation_guard_hash:" + name, _sha256(expected_path) == guard["sha256"])
                guard_body = _load(expected_path)
                check(
                    "continuation_guard_limits:" + name,
                    guard_body["status"] == "completed_within_declared_sampled_limits"
                    and guard_body["guard"]["stop_reason"] is None
                    and guard_body["guard"]["peak_swap_growth_bytes"] == 0,
                )
            check("continuation_guard_swap_zero:" + name, guard["peak_swap_growth_bytes"] == 0 and guard["peak_driver_bytes"] <= 53687091200)

    # ------------------------------------------------------------------
    # Latest vision curve verdict
    # ------------------------------------------------------------------
    curve = _load_if(root / "reports/vision_learnability_curve_v1_20261026/summary.json")
    check("vision_curve_present", curve is not None)
    if curve is not None:
        decision = curve["decision"]
        t1 = decision["t1"]
        t2 = decision["t2"]
        t3 = decision["t3"]
        check(
            "vision_curve_terminal_decision",
            curve["status"] == "curve_no_optimization_progress"
            and decision["status"] == "curve_no_optimization_progress"
            and decision["engineering_ok"] is True
            and decision["trajectory_ok"] is True,
        )
        check(
            "vision_curve_loss_reduction",
            t1["passed"] is True
            and t1["roles"]["answer"]["relative_reduction"] >= 0.25
            and t1["roles"]["caption"]["relative_reduction"] >= 0.25
            and _close(t1["roles"]["answer"]["relative_reduction"], 0.6539625409333286, 1e-12)
            and _close(t1["roles"]["caption"]["relative_reduction"], 0.6967731208816242, 1e-12),
        )
        overshoot = {}
        for role in ("answer", "caption"):
            series = t2["roles"][role]["series"]
            overshoot[role] = series[1]["loss"] > series[0]["loss"]
            check(
                "vision_curve_first_prefix_overshoot:" + role,
                overshoot[role] and t2["roles"][role]["steps"][0]["non_increasing"] is False,
            )
        check("vision_curve_monotonicity_fails", t2["passed"] is False and overshoot == {"answer": True, "caption": True})
        criteria = t3["criteria"]
        check(
            "vision_curve_zero_answer_accuracy",
            criteria["answer_correct_accuracy"]["prefix7"] == 0.0
            and criteria["answer_prefill_gold_accuracy_from_zero"]["prefix7"] == 0.0,
        )
        check(
            "vision_curve_conversion_not_established",
            criteria["answer_correct_accuracy"]["passed"] is False
            and criteria["answer_prefill_gold_accuracy_from_zero"]["passed"] is False
            and criteria["caption_token_f1_gain"]["passed"] is False
            and criteria["answer_prefill_gold_rank_mean_ratio"]["passed"] is True,
        )
        check(
            "vision_curve_no_development_or_heldout_promotion",
            decision["authorizes_v0"] is False
            and decision["v0_authorized"] is False
            and decision["dev_authorized"] is False
            and decision["dev_accessed"] is False
            and decision["heldout_authorized"] is False
            and decision["heldout_accessed"] is False
            and decision["promotion"] is False
            and decision["promotion_authorized"] is False
            and decision["retry_authorized"] is False
            and decision["no_automatic_retry"] is True,
        )
    curve_guard = _load_if(
        root / "reports/vision_learnability_curve_v1_20261026_attempt04/guard.json"
    )
    check("vision_curve_guard_present", curve_guard is not None)
    if curve_guard is not None:
        check(
            "vision_curve_guard_limits",
            curve_guard["status"] == "completed_within_declared_sampled_limits"
            and curve_guard["guard"]["stop_reason"] is None
            and curve_guard["guard"]["peak_swap_growth_bytes"] == 0
            and curve_guard["guard"]["peak_driver_bytes"] <= 53687091200,
        )
    historical_curve_guard = _load_if(
        root / "reports/vision_learnability_curve_v1_20261026/guard.json"
    )
    check("vision_curve_historical_guard_present", historical_curve_guard is not None)
    if historical_curve_guard is not None:
        check(
            "vision_curve_historical_failure_preserved",
            historical_curve_guard["status"] == "stopped_or_failed"
            and historical_curve_guard["child_exit_code"] != 0
            and historical_curve_guard["guard"]["stop_reason"] is None,
        )

    # ------------------------------------------------------------------
    # Final verification and clean-clone receipts
    # ------------------------------------------------------------------
    final_verification = _load_if(root / "reports/final_verification_20261027/summary.json")
    check("final_verification_present", final_verification is not None)
    if final_verification is not None:
        clean_export = final_verification["clean_export"]
        check("final_verification_format", final_verification["format"] == "replica.final_verification.v1")
        check(
            "final_verification_clean_clone_passed",
            clean_export["failed"] == 0
            and clean_export["errors"] == 0
            and clean_export["git_status"] == "clean"
            and clean_export["artifact_guard"]["status"] == "passed"
            and clean_export["artifact_guard"]["violations"] == [],
        )
        check(
            "final_verification_clean_clone_head",
            final_verification["head_commit"] == clean_export["head_commit"] and len(clean_export["head_commit"]) == 40,
        )
        check(
            "final_verification_workspace_failures_preserved",
            final_verification["workspace"]["failed"] == 4
            and "historical vision migration-pin" in final_verification["workspace"]["failure_scope"]
            and len(final_verification["workspace"]["failures"]) == 4,
        )
        check(
            "final_verification_no_native_job",
            final_verification["resource_and_lineage"]["native_jobs_started_by_this_verification"] == 0
            and final_verification["resource_and_lineage"]["protected_runtime_pilot_modified"] is False
            and final_verification["resource_and_lineage"]["heldout_or_sealed_final_rows_opened"] is False,
        )
    historical_clean = _load_if(root / "reports/clean_export_20261026_final/summary.json")
    check("historical_clean_export_present", historical_clean is not None)
    if historical_clean is not None:
        check(
            "historical_clean_export_zero_failures",
            historical_clean["format"] == "replica.clean_export_verification.v1"
            and historical_clean["final_attempt"]["failed"] == 0
            and historical_clean["final_attempt"]["errors"] == 0
            and historical_clean["strict_bundle"]["verified"] is True
            and historical_clean["strict_bundle"]["clean_checkout_ready"] is True,
        )
    check(
        "clean_export_receipts_present",
        (root / "reports/final_verification_20261027/clean_export/tests.xml").is_file()
        and (root / "reports/clean_export_20261026_final/tests_final.xml").is_file(),
    )

    # ------------------------------------------------------------------
    # Historical 117-input manifest remains readable and unchanged
    # ------------------------------------------------------------------
    historical_manifest_path = root / portable["historical_manifest"]["path"]
    if historical_manifest_path.is_file():
        historical = _load(historical_manifest_path)
        updates = {update["path"]: update for update in reviewed.get("historical_manifest_updates", [])}
        entries = historical["inputs"]
        historical_manifest_report = {
            "path": portable["historical_manifest"]["path"],
            "input_count": historical.get("input_count"),
            "bundle_digest": historical.get("bundle_digest"),
            "reviewed_updates": [],
            "unreviewed_drift": [],
        }
        check(
            "historical_manifest_identity",
            historical.get("input_count") == portable["historical_manifest"]["input_count"] == len(entries)
            and historical.get("bundle_digest") == portable["historical_manifest"]["bundle_digest"] == _canonical_digest(sorted(entries, key=lambda entry: entry["path"])),
        )
        for entry in entries:
            relative = entry["path"]
            path = root / relative
            if not path.is_file():
                historical_manifest_report["unreviewed_drift"].append({"path": relative, "reason": "missing"})
                check("historical_manifest_input_present:" + relative, False)
                continue
            actual = _sha256(path)
            if actual == entry["sha256"]:
                continue
            update = updates.get(relative)
            if update and update.get("prior_sha256") == entry["sha256"]:
                pinned = update.get("current_sha256")
                if pinned is not None and pinned != actual:
                    historical_manifest_report["unreviewed_drift"].append({"path": relative, "reason": "reviewed_update_hash_mismatch"})
                    check("historical_manifest_update_pin:" + relative, False)
                    continue
                historical_manifest_report["reviewed_updates"].append(
                    {"path": relative, "prior_sha256": entry["sha256"], "current_sha256": actual}
                )
                check("historical_manifest_reviewed_update:" + relative, True)
            else:
                historical_manifest_report["unreviewed_drift"].append(
                    {"path": relative, "prior_sha256": entry["sha256"], "current_sha256": actual}
                )
                check("historical_manifest_unreviewed_drift:" + relative, False)
        check("historical_manifest_no_unreviewed_drift", not historical_manifest_report["unreviewed_drift"])
    else:
        check("historical_manifest_present", False)
        check("historical_manifest_no_unreviewed_drift", False)

    # ------------------------------------------------------------------
    # Additive final-product bundle manifest
    # ------------------------------------------------------------------
    manifest_spec = portable["final_product_manifest"]
    manifest_path = root / manifest_spec["path"]
    if manifest_path.is_file():
        manifest = _load(manifest_path)
        inputs = manifest.get("inputs", [])
        declared_paths = {entry["path"] for entry in inputs}
        required_paths = set(manifest_spec["required_paths"])
        failures = []
        for entry in inputs:
            relative = entry["path"]
            if relative.startswith("/") or ".." in Path(relative).parts:
                failures.append({"path": relative, "reason": "non_relative_path"})
                continue
            path = root / relative
            if not path.is_file():
                failures.append({"path": relative, "reason": "missing"})
                continue
            if _sha256(path) != entry.get("sha256") or path.stat().st_size != entry.get("bytes"):
                failures.append({"path": relative, "reason": "hash_or_size_mismatch"})
            if not _tracked(root, relative, tracked_cache):
                failures.append({"path": relative, "reason": "untracked"})
            if not _matches_index(root, relative):
                failures.append({"path": relative, "reason": "unstaged"})
        digest = _canonical_digest(sorted(inputs, key=lambda entry: entry["path"]))
        clean_links = manifest.get("clean_export_links", [])
        linked_paths = {link.get("path") for link in clean_links if isinstance(link, dict)}
        final_manifest_report = {
            "path": manifest_spec["path"],
            "input_count": manifest.get("input_count"),
            "bundle_digest": digest,
            "failures": failures,
        }
        check(
            "final_manifest_identity",
            manifest.get("format") == "replica.sparkx_final_product_bundle.v1"
            and manifest.get("profile") == manifest_spec["profile"]
            and manifest.get("input_count") == len(inputs)
            and manifest.get("clean_checkout_ready") is True
            and manifest.get("tracking_enforced") is True,
        )
        check("final_manifest_digest", digest == manifest.get("bundle_digest"))
        check("final_manifest_inputs_match_files", not failures, failures or None)
        check("final_manifest_required_paths", required_paths <= declared_paths, sorted(required_paths - declared_paths) or None)
        check(
            "final_manifest_clean_export_links",
            set(manifest_spec["clean_export_links"]) <= linked_paths
            and linked_paths <= declared_paths,
        )
        check(
            "final_manifest_generated_receipt_not_hashed",
            manifest_spec["generated_output"] not in declared_paths,
        )
    else:
        check("final_product_bundle_manifest_present", False, manifest_spec["path"])
        final_manifest_report = {"path": manifest_spec["path"], "failures": [{"reason": "missing"}]}

    # ------------------------------------------------------------------
    # Historical verification receipt remains readable
    # ------------------------------------------------------------------
    historical_verification = _load_if(out / "verification.json")
    check("historical_verification_present", historical_verification is not None)
    check(
        "historical_verification_preserved",
        historical_verification is not None
        and historical_verification.get("checks_count") == HISTORICAL_CHECK_COUNT
        and historical_verification.get("passed") == HISTORICAL_CHECK_COUNT
        and historical_verification.get("failed") == [],
    )

    failed = [entry for entry in checks if not entry["passed"]]
    result = {
        "format": FORMAT,
        "kind": "documentation_tracked_evidence_arithmetic_verification_not_model_qualification",
        "root": str(root),
        "out": str(out),
        "status": "failed" if failed else "passed",
        "checks_count": len(checks),
        "passed": sum(entry["passed"] for entry in checks),
        "failed": failed,
        "unavailable_source_cache": unavailable,
        "source_cache_cross_checks": source_cache_cross_checks,
        "reviewed_input_changes_applied": reviewed_applied,
        "unexplained_input_drift": unexplained,
        "historical_manifest": historical_manifest_report,
        "final_product_manifest": final_manifest_report,
        "blueprint_words": len(main_text.split()),
        "native_execution": False,
        "training": False,
        "checks": checks,
    }
    if write:
        output = out / OUTPUT_NAME
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--no-write", action="store_true", help="do not write the versioned receipt")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    result = verify(args.root, args.out, write=not args.no_write)
    if not args.quiet:
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "checks_count": result["checks_count"],
                    "passed": result["passed"],
                    "failed": result["failed"][:5],
                    "unavailable_source_cache": len(result["unavailable_source_cache"]),
                    "reviewed_input_changes_applied": len(result["reviewed_input_changes_applied"]),
                    "unexplained_input_drift": result["unexplained_input_drift"],
                    "blueprint_words": result["blueprint_words"],
                },
                indent=2,
            )
        )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

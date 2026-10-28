"""Portable documentation, reference, arithmetic and migration verification.

No model, dataset, native runner or tensor library is imported.  The verifier
is deterministic for a given tree, uses repository-relative paths only, and
supports a clean checkout in which generated outputs and declared local-only
run artifacts are absent.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT_NAME = "verification.json"
FORMAT = "replica.sparkx_v5_1_review.verification.v2"
MIGRATION_FORMAT = "replica.sparkx_v5_1_review.migration.v1"
GATES_FORMAT = "replica.sparkx_v5_1_review.native_gates.v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def anchors(path: Path) -> set[str]:
    counts, result = {}, set()
    for heading in re.findall(r"^#{1,6}\s+(.+)$", path.read_text(encoding="utf-8"), re.M):
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        number = counts.get(slug, 0)
        counts[slug] = number + 1
        result.add(slug + (f"-{number}" if number else ""))
    return result


def _git_available(root: Path) -> bool:
    return (root / ".git").exists()


def _tracked(root: Path, relative: str) -> bool:
    if not _git_available(root):
        return False
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", relative],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _calculator(out: Path, root: Path):
    path = out / "calculate_capacity.py"
    spec = importlib.util.spec_from_file_location("sparkx_v5_1_capacity", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load calculator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = root
    return module.calculate()


def verify(root: Path = ROOT, out: Path = HERE, *, write: bool = True,
           tracked: str = "auto") -> dict:
    root = Path(root).resolve()
    out = Path(out).resolve()
    checks: list[dict] = []
    pending: set[str] = set()
    local_artifacts: dict[str, dict] = {}

    def check(name: str, condition, detail=None) -> None:
        row = {"name": name, "passed": bool(condition)}
        if detail is not None:
            row["detail"] = detail
        checks.append(row)

    deliverables = load(out / "deliverables.json")
    migration = load(out / "migration_manifest.json")
    snapshot = load(out / "repository_snapshot.json")
    gates = load(out / "native_gates.json")
    config = load(root / "configs/sparkx_2_6_v5_1_direction_design.json")

    # ------------------------------------------------------------------
    # Migration identity and deliverable pins
    # ------------------------------------------------------------------
    check("migration_format", migration.get("format") == MIGRATION_FORMAT)
    entries = migration.get("entries", [])
    paths = [entry.get("path") for entry in entries]
    check("migration_unique_paths", len(paths) == len(set(paths)))
    check("migration_all_reasons_present", all(entry.get("reason") for entry in entries))
    check(
        "migration_identity_digest",
        migration.get("identity", {}).get("digest_sha256")
        == hashlib.sha256(canonical(entries).encode("utf-8")).hexdigest(),
    )
    added = {e["path"] for e in entries if e.get("kind") == "added"}
    changed = {e["path"] for e in entries if e.get("kind") == "changed_existing"}
    retained = {e["path"] for e in entries if e.get("kind") == "retained"}
    check("migration_disjoint_kinds", not (added & changed) and not (added & retained) and not (changed & retained))
    generated = set(migration.get("generated_outputs", []))
    index_paths = {
        "reports/sparkx_v5_1_review_20261027/deliverables.json",
        "reports/sparkx_v5_1_review_20261027/migration_manifest.json",
    }
    local_declared = {row["path"]: row for row in migration.get("local_artifact_references", [])}

    deliverable_paths = {entry["path"] for entry in deliverables["files"]}
    check("deliverables_declared_in_migration", (deliverable_paths - index_paths) <= (added | changed | retained))

    for entry in deliverables["files"]:
        path = root / entry["path"]
        check("deliverable:" + entry["path"], path.is_file() and sha(path) == entry["sha256"])
        if path.suffix == ".md" and path.is_file():
            text = path.read_text(encoding="utf-8")
            check("fences:" + entry["path"], len(re.findall(r"^```", text, re.M)) % 2 == 0)
            for target in re.findall(r"\]\(([^\n)]+)\)", text):
                target = target.strip("<>")
                if urlparse(target).scheme:
                    continue
                destination = (path.parent / unquote(target.split("#", 1)[0])).resolve()
                label = "local_link:" + entry["path"] + ":" + target
                if destination.is_relative_to(root):
                    relative = destination.relative_to(root).as_posix()
                else:
                    relative = None
                if not destination.exists():
                    if relative in generated:
                        check("generated_output_absent:" + relative, True)
                    elif relative in local_declared:
                        check("local_artifact_absent:" + relative, True,
                              "declared unversioned local evidence; not required in a clean checkout")
                        local_artifacts[relative] = {"present": False}
                    else:
                        check(label, False, "missing local link target")
                    continue
                check(label, True)
                if destination.is_dir():
                    check("link_target_directory:" + entry["path"] + ":" + target, True)
                    continue
                if destination.is_relative_to(root):
                    if relative in local_declared:
                        row = local_declared[relative]
                        if destination.is_file():
                            check("local_artifact_hash:" + relative, sha(destination) == row["sha256"])
                            local_artifacts[relative] = {"present": True, "hash_matches": sha(destination) == row["sha256"]}
                        check("local_artifact_not_required:" + relative, row.get("required_in_clean_checkout") is False)
                    elif relative in generated:
                        check("generated_output_present:" + relative, False, "generated output must not be committed evidence")
                    elif _git_available(root):
                        if _tracked(root, relative):
                            pass
                        elif relative in added or relative in index_paths:
                            pending.add(relative)
                            check("link_target_declared_new:" + relative, True)
                            if tracked == "require":
                                check("link_target_tracked_after_commit:" + relative, False)
                        else:
                            check("link_target_untracked:" + relative, False, target)
                if "#" in target and destination.is_file() and destination.suffix == ".md":
                    fragment = unquote(target.split("#", 1)[1])
                    check("section_link:" + entry["path"] + ":" + target, fragment in anchors(destination))

    # ------------------------------------------------------------------
    # Preserved inputs, reviewed changes and retention
    # ------------------------------------------------------------------
    by_path = {entry["path"]: entry for entry in entries}
    snapshot_shas = {row["path"]: row["sha256"] for row in snapshot["inputs"]}
    preserved = snapshot["inputs"] + [snapshot["attachment"]]
    for row in preserved:
        path = root / row["snapshot"]
        check("preserved_input:" + row["snapshot"], path.is_file() and sha(path) == row["sha256"])

    unexplained: list[dict] = []
    for row in snapshot["inputs"]:
        relative = row["path"]
        path = root / relative
        if not path.is_file():
            unexplained.append({"path": relative, "reason": "missing_snapshot_input"})
            continue
        actual = sha(path)
        entry = by_path.get(relative)
        if actual == row["sha256"]:
            if entry is not None:
                check("retained_identity_unchanged:" + relative, entry.get("kind") == "retained")
            continue
        if (
            entry is not None
            and entry.get("kind") == "changed_existing"
            and entry.get("old", {}).get("sha256") == row["sha256"]
            and entry.get("new", {}).get("sha256") == actual
        ):
            check("reviewed_old_new_identity:" + relative, True, entry["old"].get("identity"))
        else:
            unexplained.append({"path": relative, "reviewed_sha256": row["sha256"], "current_sha256": actual})
            check("unexplained_input_drift:" + relative, False, actual)
    check("no_unexplained_snapshot_drift", not unexplained, unexplained or None)

    for entry in entries:
        path = root / entry["path"]
        kind = entry.get("kind")
        if kind == "added":
            check("added_identity_present:" + entry["path"], path.is_file() and sha(path) == entry.get("new", {}).get("sha256"))
            check("added_not_historical:" + entry["path"], entry["path"] not in {row["path"] for row in snapshot["inputs"]})
            check("added_identity_named:" + entry["path"], bool(entry.get("new", {}).get("identity")))
            if _git_available(root) and not _tracked(root, entry["path"]):
                pending.add(entry["path"])
        elif kind == "changed_existing":
            check("changed_identity_present:" + entry["path"], path.is_file() and sha(path) == entry.get("new", {}).get("sha256"))
            check("migration_old_matches_snapshot:" + entry["path"],
                  entry["path"] not in snapshot_shas
                  or entry.get("old", {}).get("sha256") == snapshot_shas[entry["path"]])
            check("changed_old_new_differ:" + entry["path"], entry.get("old", {}).get("sha256") != entry.get("new", {}).get("sha256"))
            check("changed_identity_named:" + entry["path"],
                  bool(entry.get("old", {}).get("identity")) and bool(entry.get("new", {}).get("identity")))
        elif kind == "retained":
            current = sha(path) if path.is_file() else None
            check("retained_identity_present:" + entry["path"],
                  current == entry.get("old", {}).get("sha256") == entry.get("new", {}).get("sha256"))
        else:
            check("migration_known_kind:" + str(entry.get("path")), False, kind)

    for relative, row in local_declared.items():
        path = root / relative
        check("local_artifact_declared_once:" + relative, bool(row.get("reason")) and row.get("required_in_clean_checkout") is False)
        if path.is_file():
            check("local_artifact_pinned:" + relative, sha(path) == row.get("sha256"))
            local_artifacts[relative] = {"present": True, "hash_matches": sha(path) == row.get("sha256")}
        else:
            check("local_artifact_may_be_absent:" + relative, True)
            local_artifacts[relative] = {"present": False}

    # Historical controls and retained identifiers
    control = next(x for x in snapshot["inputs"] if x["path"].endswith("final_product_v5.json"))
    check("historical_v5_configuration_unchanged", sha(root / control["path"]) == control["sha256"])
    check("historical_v5_control_retained", by_path.get(control["path"], {}).get("kind") == "retained")
    retained_names = migration.get("retained_identifiers", [])
    joined = "\n".join((root / p).read_text(encoding="utf-8") for p in deliverable_paths
                       if (root / p).suffix == ".md")
    check("retained_identifiers_named", bool(retained_names) and all(name in joined for name in retained_names))

    # ------------------------------------------------------------------
    # Config/report references and JSON integrity
    # ------------------------------------------------------------------
    for name, relative in config["references"].items():
        path = root / relative
        check("config_reference_present:" + name, path.is_file())
        if name == "historical_control":
            check("config_reference_historical_pin:" + name, sha(path) == control["sha256"])
        else:
            check("config_reference_declared:" + name, relative in (added | changed | retained | index_paths))
    check("native_gates_referenced", config["references"].get("native_gates") == "reports/sparkx_v5_1_review_20261027/native_gates.json")

    for path in sorted(out.glob("*.json")):
        if path.name == OUTPUT_NAME:
            continue
        try:
            data = load(path)
        except (OSError, ValueError):
            check("report_json_parses:" + path.name, False)
            continue
        check("report_json_parses:" + path.name, isinstance(data, (dict, list)))
    check("deliverables_date", deliverables.get("date") == "2026-10-27")

    # ------------------------------------------------------------------
    # Sources, claims and bibliography inventory
    # ------------------------------------------------------------------
    sources = load(out / "evidence.json")["sources"]
    check("23_primary_records", len(sources) == 23)
    check("18_fresh_records", sum(x["review_type"] == "fresh_primary_source" for x in sources) == 18)
    for source in sources:
        check("source:" + source["name"], source["retrieval_status"] == "success" and
              bool(re.fullmatch(r"[0-9a-f]{64}", source["sha256"])) and
              source["url"].startswith("https://") and source["bytes"] > 0 and
              bool(source["reviewed_sections"]) and bool(source["interpretation_limit"]))
    check("unique_source_names", len({x["name"] for x in sources}) == len(sources))

    claims = load(out / "claim_dispositions.json")
    check("all_16_proposals_disposed", {x["id"] for x in claims} == {f"C{i}" for i in range(1, 17)} and len(claims) == 16)
    inventory = load(out / "attachment_reference_inventory.json")
    check("attachment_inventory_count", inventory["count"] == len(inventory["entries"]) == 109)
    check("unreviewed_references_not_promoted", all(x["status"] == "not_revalidated_in_this_review"
                                                    for x in inventory["entries"] if not x["reviewed_source_ids"]))

    # ------------------------------------------------------------------
    # Declarative design, staging and precision identity
    # ------------------------------------------------------------------
    check("design_only_not_execution", config["status"] == "design_only" and config["executable"] is False
          and config["runtime_admission_changed"] is False)
    check("offline_modality_training", config["offline_multimodal_training"] == "full_finetuning"
          and config["F2_online_memory_updates"] == 0)
    check("NaFlex_default_vs_override", config["vision"]["processor_default_max_patches"] == 256
          and config["vision"]["candidate_override_max_patches"] == 1024)
    check("historical_one_step_preserved", config["plastic"]["historical_one_clock_steps_per_event"] == 1)
    check("product_key_cartesian_size", config["Mkv"]["subkeys_per_table"] ** 2 == config["Mkv"]["rows"])
    check("Mkv_not_cache_namespace", config["Mkv"]["sparse_optimizer_qualified"] is False)
    for name, profile in config["plastic"]["profiles"].items():
        check("pool_partition:" + name, profile["fast"] + profile["slow"] == profile["experts"])
        check("hot_record_bound:" + name, profile["hot_records"] <= profile["records"])

    precision = config["precision_identities"]
    check("precision_identities_distinct", set(precision) == {
        "official_bf16", "official_fp8_storage", "dequantized_fp8_derivative",
        "native_fp8_execution", "training_stack"})
    check("official_fp8_not_executed", precision["official_fp8_storage"]["native_execution"] is False
          and precision["official_fp8_storage"]["execution_status"] == "refused_on_mps_never_executed")
    derivative = precision["dequantized_fp8_derivative"]
    check("derivative_is_not_official", derivative["storage_dtype"] == "BF16"
          and derivative["per_token_f8_quantization"] is False
          and derivative["is_official_bf16_release"] is False
          and derivative["is_official_fp8_storage"] is False)
    check("native_fp8_not_executed", precision["native_fp8_execution"]["executed"] is False
          and precision["native_fp8_execution"]["status"] == "never_executed_refused_on_mps")
    check("no_precision_full_training_claim", all(
        value["full_training_qualified"] is False
        for name, value in precision.items() if name != "training_stack"))
    check("training_fp32_ownership", precision["training_stack"]["parameter_and_optimizer_state_ownership"] == "FP32"
          and precision["training_stack"]["full_training_qualified"] is False)

    staging = config["staging"]
    order = staging["order"]
    check("staging_order_small_learner_first", order[0] == "small_useful_learner")
    check("staging_P32_after_small_learner", order.index("P32") > order.index("small_useful_learner"))
    check("staging_P64_after_P32", order.index("P64") > order.index("P32"))
    check("staging_Mkv_after_P", order.index("Mkv_4k_smoke") > order.index("P32")
          and order.index("Mkv_64k") > order.index("Mkv_4k_smoke"))
    check("staging_P_plus_Mkv_last", order[-1] == "P_plus_Mkv")
    check("sparse_preconditions_complete", all(staging["sparse_preconditions"].values()))
    lineage = staging["data_lineage"]
    check("data_lineage_source_groups", len(lineage["split_units"]) >= 5
          and lineage["derived_examples_inherit_source_split"] is True)
    check("audit_not_recycled", lineage["audit_sealed_before_selection"] is True
          and lineage["audit_recycling_prohibited"] is True
          and lineage["development_commit_probes_are_not_heldout_proof"] is True)

    ladder = config["learning_dependency_order"]
    ids = [row["id"] for row in ladder]
    check("learning_ladder_orders", [row["order"] for row in ladder] == list(range(1, len(ladder) + 1)))
    positions = {row["id"]: row["order"] for row in ladder}
    check("teacher_before_outer", positions["teacher_competence_and_work_declaration"]
          < positions["outer_loop_training"])
    check("gradient_before_outer", positions["gradient_approximation_declaration"]
          < positions["outer_loop_training"])
    check("teacher_before_all_later", positions["teacher_competence_and_work_declaration"]
          < min(positions["multiple_clocks"], positions["self_distillation"], positions["rsi_policy_improvement"]))
    check("self_distillation_after_clocks", positions["self_distillation"] > positions["multiple_clocks"])
    check("rsi_last", ids[-1] == "rsi_policy_improvement")
    available: set[str] = set()
    unresolved_requires = []
    for node in ladder:
        for requirement in node["requires"]:
            if requirement not in available:
                unresolved_requires.append({"node": node["id"],
                                            "requires": requirement})
        available |= ({node["id"]} | set(node["declares"])
                      | set(node.get("provides", [])))
    check("learning_dependency_requires_resolved", not unresolved_requires,
          unresolved_requires or None)
    contract = config["dependency_contract"]
    check("learning_dependency_sequence",
          ids == contract["required_sequence"]
          and [node["order"] for node in ladder] == list(range(1, len(ladder) + 1)))
    gate_rows = {gate["id"]: gate for gate in gates.get("gates", [])}
    referenced_gates = {gate for refs in contract["native_gate_refs"].values()
                        for gate in refs}
    check("learning_dependency_gate_refs",
          referenced_gates <= set(gate_rows) and {"NG6", "NG7"} <= referenced_gates)
    check("learning_dependency_gates_unresolved",
          all(gate_rows[gate]["status"] == "unresolved"
              and gate_rows[gate]["executed"] is False
              and gate_rows[gate]["measured"] is None
              for gate in referenced_gates))
    rs_gates = contract["rs_gates"]
    check("rsi_gates_complete",
          [row["id"] for row in rs_gates] == [f"RS{i}" for i in range(7)])
    check("rsi_gates_unresolved",
          all(row["status"] == "unresolved" for row in rs_gates))
    check("learning_dependency_no_promotion",
          contract["gate_status"] == "unresolved_not_promoted"
          and contract["promotion"] == "none")
    check("personality_outside_ladder",
          "personality" not in ids
          and contract["personality"]
          == "separate_pending_not_in_learning_dependency_order")
    check("teacher_work_items_declared", {"teacher_forward_positions", "student_rollouts", "replay_passes",
                                          "evaluation_positions", "target_checking_work"} <= set(
        ladder[1]["declares"]))
    check("personality_separate", config["personality"]["specified_separately"] is True
          and config["personality"]["evaluated_separately"] is True
          and config["personality"]["inferred_from_research_or_affect_labels"] is False)

    # ------------------------------------------------------------------
    # Analytical arithmetic and residency inventory
    # ------------------------------------------------------------------
    expected = {
        "mini_P32": (25165824, 288, 24.5, 4194304),
        "spark_P64": (50331648, 576, 130, 16777216),
        "spark_P128_stretch": (100663296, 1152, 130, 16777216),
    }
    a = _calculator(out, root)
    check("ledger_reproduces", a == load(out / "analytical_capacity.json"))

    reconciliation = a["precision_byte_reconciliation"]
    check("precision_byte_identities", set(reconciliation) == {
        "official_bf16", "official_fp8_storage",
        "dequantized_fp8_derivative", "native_fp8_execution"})
    check("precision_byte_reconciled",
          all(row["reconciled"] is True for row in reconciliation.values()))
    bf16_bytes = reconciliation["official_bf16"]
    check("precision_bf16_bytes",
          bf16_bytes["unique_parameter_elements"] == 4112079360
          and bf16_bytes["tensor_count"] == 290
          and bf16_bytes["payload_bytes"] == 8224158720
          and bf16_bytes["payload_mib"] == 7843.1689453125)
    fp8_bytes = reconciliation["official_fp8_storage"]
    check("precision_fp8_bytes",
          fp8_bytes["tensor_count"] == 470
          and fp8_bytes["quantized_tensor_count"] == 180
          and fp8_bytes["quantized_parameter_elements"] == 3774873600
          and fp8_bytes["scale_tensor_count"] == 180
          and fp8_bytes["scale_tensor_elements"] == 1142784
          and fp8_bytes["bf16_payload_bytes"] == 676697088
          and fp8_bytes["f8_e4m3_payload_bytes"] == 3774873600
          and fp8_bytes["payload_bytes"] == 4451570688
          and fp8_bytes["payload_mib"] == 4245.3486328125
          and fp8_bytes["bytes_per_parameter_effective"] != 1)
    derivative_bytes = reconciliation["dequantized_fp8_derivative"]
    check("precision_derivative_bytes",
          derivative_bytes["payload_bytes"] == 8224158720
          and derivative_bytes["source_fp8_payload_bytes"] == 4451570688
          and derivative_bytes["serialized_file_bytes"] == 8224192264
          and derivative_bytes["payload_bytes"] != derivative_bytes["source_fp8_payload_bytes"])
    check("precision_native_fp8_no_invented_bytes",
          reconciliation["native_fp8_execution"]["residency_bytes"] is None)
    declared_order = a["dependency_order"]
    check("learning_dependency_ledger",
          declared_order["sequence"] == config["dependency_contract"]["required_sequence"]
          and declared_order["all_requires_resolved"] is True)
    check("E2_fields_no_duplicate_old_key", a["record_bytes"] == {"compact": 3328, "hot": 32768, "entity": 896})
    for name, (parameters, live, episodic, query_mac) in expected.items():
        value = a["profiles"][name]
        check("profile_arithmetic:" + name, (value["P_parameters"], value["P_live_mib"],
                                             value["E2_payload_mib"], value["two_global_dense_query_MAC"])
              == (parameters, live, episodic, query_mac))
        check("two_expert_staging:" + name, value["selected_two_expert_weight_moment_gradient_staging_mib"] == 24)
        check("not_measured_fit:" + name, value["measured_fit"] is False)
    check("equal_capacity_wide_control", a["wide_P16_parameters"] == 50331648)
    check("sparse_dense_compute", a["P_expert_matrix_MAC_per_position_at_9_sites"] ==
          {"2": 14155776, "4": 28311552, "16": 113246208, "64": 452984832})
    check("Mkv_values_and_moments", a["Mkv"]["value_parameters"] == 16777216 and a["Mkv"]["value_live_mib"] == 192)
    check("Mkv_staging", a["Mkv"]["selected_row_weight_moment_gradient_staging_mib"] == 1)
    check("Mkv_key_and_query_geometry", a["Mkv"]["fixed_product_key_parameters"] == 32768
          and a["Mkv"]["joint_candidates"] == 64)
    check("Mkv_1M_growth_cost", a["Mkv_growth"]["1048576"] == {
        "value_parameters": 268435456, "FP32_values_two_moments_mib": 3072,
        "subkeys_per_half": 1024, "subkey_score_MAC_per_query": 131072})
    check("reader_KV_payload", a["projected_reader_KV_mib"] == {"7": 2.734375, "9": 3.515625})
    check("nonlinear_F", a["nonlinear_F_candidate"]["parameters"] == 6291456)
    check("AV_positions", a["AV_16s_example_positions"] == {"coarse": 1320, "two_extra_crops": 1832})
    check("fixed_384_patch14_counterexample", ((384 // 14 + 1) // 2) ** 2 == 196)

    routes = a["activation_workloads"]["routes"]
    check("activation_workloads_not_measured",
          a["activation_workloads"]["measured"] is False)
    check("activation_old_base_route",
          routes["old_base_vision"]["width"] == 768
          and routes["old_base_vision"]["outputs"] == 196
          and routes["old_base_vision"]["elements_per_view_per_layer"] == 150528
          and routes["old_base_vision"]["bf16_bytes_per_view_per_layer"] == 301056)
    check("activation_candidate_default_route",
          routes["candidate_naflex_default"]["width"] == 1152
          and routes["candidate_naflex_default"]["outputs"] == 256
          and routes["candidate_naflex_default"]["elements_per_view_per_layer"] == 294912
          and routes["candidate_naflex_default"]["bf16_bytes_per_view_per_layer"] == 589824)
    check("activation_candidate_override_route",
          routes["candidate_naflex_override_1024"]["merge"] == [2, 2]
          and routes["candidate_naflex_override_1024"]["patches"] == 1024
          and routes["candidate_naflex_override_1024"]["outputs"] == 256
          and routes["candidate_naflex_override_1024"][
              "pre_merge_elements_per_view_per_layer"] == 1179648)
    counterexample = routes["candidate_fixed_384_patch14_counterexample"]
    check("activation_fixed_384_counterexample",
          counterexample["valid_patches"] == 729
          and counterexample["padded_patches"] == 784
          and counterexample["outputs"] == 196
          and counterexample["expected_256"] is False)

    inventory_arith = a["residency_inventory"]
    components = inventory_arith["components"]
    check("residency_not_measured_fit", inventory_arith["measured_fit"] is False
          and inventory_arith["full_training_qualified"] is False)
    expected_components = {
        "base_backbone_bf16_storage": 7843.1689453125,
        "base_backbone_fp32_weights": 15686.337890625,
        "base_backbone_fp32_gradients": 15686.337890625,
        "base_backbone_fp32_adam_moments": 31372.67578125,
        "base_vision_siglip2_base_bf16_storage": 177.16259765625,
        "base_vision_siglip2_base_fp32_weights": 354.3251953125,
        "base_vision_siglip2_base_fp32_gradients": 354.3251953125,
        "base_vision_siglip2_base_fp32_adam_moments": 708.650390625,
        "base_audio_spear_declared_bf16_storage": 177.93555450439453,
        "base_audio_spear_fp32_weights": 355.87110900878906,
        "base_audio_spear_fp32_gradients": 355.87110900878906,
        "base_audio_spear_fp32_adam_moments": 711.7422180175781,
        "vision_projector_1152_1024_2560": 43.541015625,
        "vision_projector_1152_1024_2560_fp32_gradients": 14.513671875,
        "audio_projector_512_1024_2560": 36.041015625,
        "audio_projector_512_1024_2560_fp32_gradients": 12.013671875,
        "P16_historical": 144.0,
        "P32": 288.0,
        "P64": 576.0,
        "P128_stretch": 1152.0,
        "Mkv_4k_values_and_moments": 12.0,
        "Mkv_64k_values_and_moments": 192.0,
        "Mkv_fixed_product_keys": 0.125,
        "Mkv_nine_query_projections": 1.125,
        "selected_two_expert_update_staging": 24.0,
        "selected_256_row_Mkv_staging": 1.0,
    }
    for name, mib in expected_components.items():
        row = components.get(name, {})
        check("residency_component:" + name, row.get("mib") == mib)
        check("residency_not_qualified:" + name,
              row.get("full_training_qualified") is False
              and row.get("measured_fit") is False)
    check("residency_evidence_classes",
          {row["evidence_class"] for row in components.values()}
          <= {"measured_native", "analytical_estimate", "declared_metadata"})
    check("residency_measured_only_sealed_sources",
          [name for name, row in components.items()
           if row["evidence_class"] == "measured_native"]
          == ["base_backbone_bf16_storage",
              "base_vision_siglip2_base_bf16_storage"])
    check("residency_vision_tensor_inventory",
          components["base_vision_siglip2_base_bf16_storage"]["parameter_elements"] == 92884224
          and components["base_vision_siglip2_base_bf16_storage"]["tensor_count"] == 208)
    check("residency_audio_declared_inventory",
          components["base_audio_spear_declared_bf16_storage"]["parameter_elements"] == 93289476
          and components["base_audio_spear_declared_bf16_storage"]["evidence_class"]
          == "declared_metadata"
          and bool(components["base_audio_spear_declared_bf16_storage"][
              "missing_prerequisite"]))
    check("residency_patch_widths",
          config["vision"]["base_control"]["hidden_size"] == 768
          and config["vision"]["hidden_size"] == 1152)
    check("residency_training_gradients_staged",
          all(components[name]["bytes_per_element"] == 4
              and components[name]["trainable_in_offline_full_finetuning"] is True
              for name in (
                  "base_vision_siglip2_base_fp32_gradients",
                  "base_audio_spear_fp32_gradients",
                  "vision_projector_1152_1024_2560_fp32_gradients",
                  "audio_projector_512_1024_2560_fp32_gradients"))
          and inventory_arith["training_contract"]["stages"][
              "offline_full_finetuning"]["bytes_per_trainable_parameter"] == 16)
    unsealed = inventory_arith["unsealed"]
    check("residency_unsealed_named", {"retained_vision_tower_so400m", "activations",
                                       "restart_transient_full_checkpoint_copy"} <= set(unsealed))
    check("residency_so400m_unsealed",
          unsealed["retained_vision_tower_so400m"]["evidence_class"] == "unsealed"
          and "no local So400m NaFlex tensor index" in (
              unsealed["retained_vision_tower_so400m"]["missing_prerequisite"] or ""))
    sealed = inventory_arith["sealed_source_inventory"]
    check("residency_sealed_inventory",
          sealed["base_vision_siglip2"]["vision_tensor_count"] == 208
          and sealed["base_vision_siglip2"]["parameter_elements"] == 92884224
          and sealed["base_audio_spear"]["parameter_elements"] == 93289476
          and sealed["candidate_so400m_naflex"]["status"]
          == "unsealed_missing_local_tensor_index")
    check("residency_feasibility_not_weight_only",
          inventory_arith["feasibility"]["weight_only_feasibility_claim"] is False
          and inventory_arith["feasibility"]["weight_only_total_mib"] is None
          and inventory_arith["feasibility"]["gate_status"] == "unresolved")
    check("residency_restart_transients", inventory_arith["restart_transients"]["selected_two_expert_update_staging_mib"] == 24
          and inventory_arith["restart_transients"]["selected_256_row_Mkv_staging_mib"] == 1)

    # ------------------------------------------------------------------
    # Native gates stay separate and unresolved
    # ------------------------------------------------------------------
    check("native_gates_format", gates.get("format") == GATES_FORMAT)
    gate_rows = gates.get("gates", [])
    check("native_gates_unique", len({gate["id"] for gate in gate_rows}) == len(gate_rows))
    check("native_gates_not_executed", all(gate["status"] == "unresolved"
                                           and gate["executed"] is False
                                           and gate["measured"] is None for gate in gate_rows))
    summary = gates.get("summary", {})
    check("native_gates_summary", summary.get("total") == len(gate_rows)
          and summary.get("unresolved") == sum(1 for gate in gate_rows if gate["status"] == "unresolved")
          and summary.get("executed") == sum(1 for gate in gate_rows if gate["executed"])
          and summary.get("passed") == sum(1 for gate in gate_rows if gate["status"] == "passed"))

    # ------------------------------------------------------------------
    # Authored content assertions
    # ------------------------------------------------------------------
    blueprint = (root / "docs/architecture/23_SparkX_2_6_4B_Final_Product_Blueprint.md").read_text(encoding="utf-8")
    check("blueprint_current_version", "**Version 5.1" in blueprint)
    check("learning_and_recall_not_conflated", "148/152" in blueprint and "0/28" in blueprint and "host selection" in blueprint)
    check("full_total_not_invented", "No final stable subtotal is sealed for v5.1" in blueprint)
    check("state_namespace_and_checkpoint", "mkv_values_and_optimizer.safetensors" in blueprint)
    plan = (root / "docs/architecture/24_SparkX_2_6_v5_1_Refinement_Plan.md").read_text(encoding="utf-8")
    check("plan_residency_inventory_section", "### 4.4 Residency inventory" in plan
          and "### 4.5 Staged admission" in plan)
    check("plan_precision_identities", "native FP8 execution" in plan and "derivative_02" in plan)
    mini = (root / "docs/architecture/25_MiniCPM5_Direction_Correction_and_Learning_Capacity_Plan.md").read_text(encoding="utf-8")
    check("mini_sparse_contract", "row-sparse contract" in mini and "duplicate-index" in mini)
    check("mini_declared_work_order", "Teacher competence, gradient approximation" in mini)

    if tracked == "require" and _git_available(root):
        for relative in sorted((added | deliverable_paths) - index_paths - generated):
            check("bundle_tracked:" + relative, _tracked(root, relative))

    failed = [row for row in checks if not row["passed"]]
    result = {
        "format": FORMAT,
        "scope": "documentation, reference, arithmetic and migration checks only; no model, training, native execution or measured fit",
        "checks": len(checks),
        "passed": len(checks) - len(failed),
        "failed": failed,
        "details": checks,
        "migration": {
            "format": migration.get("format"),
            "identity": migration.get("identity", {}).get("id"),
            "digest_sha256": migration.get("identity", {}).get("digest_sha256"),
            "changed": sorted(changed),
            "added": sorted(added),
            "retained": sorted(retained),
        },
        "local_artifacts": dict(sorted(local_artifacts.items())),
        "pending_commit_additions": sorted(pending),
        "native_gates": {
            "file": "native_gates.json",
            "total": len(gate_rows),
            "executed": sum(1 for gate in gate_rows if gate["executed"]),
            "unresolved": [gate["id"] for gate in gate_rows if gate["status"] == "unresolved"],
            "note": "native measurement is not performed or inferred by this verifier",
        },
    }
    if write:
        (out / OUTPUT_NAME).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("scope", "checks", "passed")}
                     | {"failed": len(failed), "migration_identity": result["migration"]["identity"],
                        "native_gates_unresolved": len(result["native_gates"]["unresolved"])}, indent=2))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--tracked", choices=("auto", "require", "skip"), default="auto")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    result = verify(root=args.root, out=args.out, write=not args.no_write, tracked=args.tracked)
    return 0 if not result["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

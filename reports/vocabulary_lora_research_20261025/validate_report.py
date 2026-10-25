"""Validate research artifacts only; does not run or admit a model.

Optional --source-dir verifies temporary public-source snapshots.
Optional --tokenizer reruns the text-only probe from a hash-checked JSON file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

from estimate_costs import calculate
from estimate_swappable_costs import calculate_swappable


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    args = parser.parse_args()
    report_dir = Path(__file__).resolve().parent
    root = report_dir.parents[1]
    docs = root / "docs/research/vocabulary_lora_20261025"
    output = report_dir / "validation.json"
    failures: list[str] = []
    checks: list[str] = []

    def require(condition: bool, name: str) -> None:
        (checks if condition else failures).append(name)

    plan = json.loads((root / "configs/vocabulary_lora_research_plan_v1.json").read_text())
    review = json.loads((report_dir / "review.json").read_text())
    costs = json.loads((report_dir / "analytical_costs.json").read_text())
    probe = json.loads((report_dir / "tokenizer_probe.json").read_text())
    swap_review = json.loads((report_dir / "swappable_review.json").read_text())
    swap_costs = json.loads((report_dir / "swappable_costs.json").read_text())
    require(swap_costs == calculate_swappable(), "swappable_costs_recomputed_exactly")
    encoder = swap_costs["illustrative_shared_phrase_encoder"]
    require(encoder["matrix_parameters"] == 2883584
            and encoder["bf16_matrix_weights_mib"] == 5.5
            and encoder["fp32_weights_grad_two_adam_moments_mib"] == 44,
            "small_phrase_encoder_independent_budget_anchor")
    require(swap_costs["active_rows"][0]["tied_bf16_rows_mib"] == 1.25
            and swap_costs["active_rows"][-1]["two_full_row_buffers_mib"] == 10,
            "swappable_row_budget_anchors")
    require(costs == json.loads(json.dumps(calculate())), "costs_recomputed_exactly")
    require(costs["spark"]["base_tied_embedding_bf16_mib"] == 640, "base_embedding_640_mib")
    row = next(x for x in costs["vocabulary"] if x["new_rows"] == 1024)
    require(row["parameters"] == 2621440 and row["bf16_weights_mib"] == 5
            and row["fp32_weights_grad_two_adam_moments_mib"] == 40,
            "1024_row_budget_independent_anchor")
    all_layers = next(x for x in costs["lora"]
                      if x["scope"] == "all_36_blocks_five_projections" and x["rank"] == 8)
    require(all_layers["total_parameters"] == 15482880
            and all_layers["fp32_weights_grad_two_adam_moments_mib"] == 236.25,
            "rank8_all_layers_independent_anchor")
    require(costs["saved_activation_examples"][0]["care_fp32_z_and_m_bytes"] == 163904
            and costs["saved_activation_examples"][2]["care_fp32_z_and_m_bytes"] == 180224,
            "care_short_and_long_batch_payload_anchors")

    cfg = json.loads((root / "reports/spark_preparation_20261025/run_01/assets/official_bf16_control/config.json").read_text())
    require(all(cfg[k] == costs["spark"][k] for k in
                ("hidden_size", "vocab_size", "num_hidden_layers", "tie_word_embeddings")),
            "dimensions_match_prepared_spark_source")
    require(not plan["architecture_implemented_by_this_review"] and not plan["model_training_performed"],
            "proposal_status_is_explicit")
    require(plan["vocabulary_extension"]["first_new_id"] == cfg["vocab_size"]
            and not plan["vocabulary_extension"]["reserved_id_reuse"]
            and not plan["vocabulary_extension"]["online_id_recycling"], "append_only_identity_contract")
    swap = plan["swappable_dynamic_vocabulary"]
    require(swap["physical_slot_reuse"] and not swap["permanent_token_id_reassignment"]
            and swap["full_base_vocabulary_retained"], "temporary_and_permanent_identity_separated")
    require(swap["initial_mapping_lifetime"] == "One generation"
            and not swap["row_is_independently_optimized_slot_parameter"]
            and swap["native_selector_required"], "initial_swap_lifetime_and_native_learning_contract")

    stages = {s["id"]: s for s in plan["stages"]}
    gates = {g["id"]: g for g in plan["gates"]}
    require(len(stages) == len(plan["stages"]) == 9, "nine_unique_stages")
    require(len(gates) == len(plan["gates"]) == 12, "twelve_unique_gates")
    require(all(d in stages for s in stages.values() for d in s["depends_on"]),
            "all_stage_dependencies_exist")
    require(all(g in gates for s in stages.values() for g in s["gates"]),
            "all_stage_gates_exist")
    remaining = dict(stages)
    ordered = []
    while remaining:
        ready = [k for k, v in remaining.items() if set(v["depends_on"]) <= set(ordered)]
        if not ready:
            break
        for k in ready:
            ordered.append(k)
            del remaining[k]
    require(not remaining, "stage_graph_is_acyclic")
    require(all((root / x).is_file() for x in plan["inherits"]), "inherited_plans_exist")

    require(probe["vocabulary"]["all_entries"] == 131072
            and probe["vocabulary"]["max_id"] == 131071
            and probe["vocabulary"]["ids_contiguous"], "tokenizer_inventory")
    require(len(probe["baseline_examples"]) == 11
            and all(x["roundtrip"] for x in probe["baseline_examples"]),
            "eleven_baseline_roundtrips")
    trials = probe["temporary_in_memory_addition_trials"]
    require(len(trials) == 3 and all(x["new_id"] == 131072
                                    and x["old_id_mapping_preserved"] for x in trials),
            "temporary_additions_preserve_old_ids")
    require(all(c["roundtrip"] for t in trials for c in t["cases"]),
            "all_addition_trial_roundtrips")
    require(probe["baseline_examples"][2]["token_count"] == 20
            and trials[0]["cases"][0]["token_count"] == 14
            and trials[1]["cases"][0]["token_count"] == 20,
            "illustrative_compression_and_boundary_effect")
    if args.tokenizer:
        from probe_tokenizer import probe as rerun_probe
        require(rerun_probe(args.tokenizer) == probe, "text_only_probe_reproduced_exactly")

    texts = {p: p.read_text() for p in docs.glob("*.md")}
    require(len(texts) == 6, "six_research_documents")
    local_link_count = 0
    for path, content in texts.items():
        require(sum(line.startswith("```") for line in content.splitlines()) % 2 == 0,
                "balanced_fences:" + path.name)
        for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", content):
            target = match.group(1).strip().strip("<>")
            if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target) or target.startswith("#"):
                continue
            target = unquote(target.split("#", 1)[0])
            resolved = (path.parent / target).resolve()
            local_link_count += 1
            require(resolved.exists() or resolved == output, "local_link:" + path.name + ":" + target)
    all_text = "\n".join(texts.values())
    papers = [s for s in review["primary_sources"] if s["kind"] == "research_paper"]
    require(len(papers) == len({s["arxiv_version"].split("v")[0] for s in papers}) == 24,
            "24_distinct_versioned_papers")
    require(all(s["version_url"] in all_text for s in papers), "all_papers_linked_in_documents")
    additional = swap_review["primary_sources"]
    require(len(additional) == 4
            and len({s["arxiv_version"].split("v")[0] for s in papers + additional}) == 28,
            "four_additional_distinct_swappable_papers")
    require(all(s["version_url"] in all_text for s in additional), "all_swappable_papers_linked")
    sources = review["primary_sources"] + additional
    require(all(re.fullmatch(r"[0-9a-f]{64}", s["sha256"])
                and s["bytes"] > 0 and s["reading_scope"] for s in sources),
            "all_evidence_has_hash_size_and_reading_scope")
    require(all(re.fullmatch(r"[0-9a-f]{40}", s["revision"])
                for s in review["official_repositories"]), "official_code_revisions_pinned")
    require(all((root / x).exists() or root / x == output for x in review["deliverables"]),
            "declared_deliverables_exist")

    source_hash_checks = 0
    if args.source_dir:
        for source in sources:
            p = args.source_dir / (source["name"] + ".html")
            valid = p.is_file() and len(p.read_bytes()) == source["bytes"]
            if valid:
                valid = hashlib.sha256(p.read_bytes()).hexdigest() == source["sha256"]
            require(valid, "source_snapshot:" + source["name"])
            source_hash_checks += 1

    changed_since_snapshot = []
    for name, digest in review["completion_repository_snapshot"]["files"].items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            changed_since_snapshot.append(name)
    # Concurrent runtime work is recorded, not claimed to be validated here.
    result = {
        "schema": "replica.vocabulary_lora_research_validation.v1",
        "status": "pass" if not failures else "fail",
        "scope": "Research artifact integrity, tokenizer-only probe and analytical arithmetic; not model admission",
        "repository_head_observed": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "checks_passed": len(checks), "failures": failures,
        "stage_topological_order": ordered, "local_links_checked": local_link_count,
        "public_source_hashes_checked": source_hash_checks,
        "tokenizer_probe_rerun": bool(args.tokenizer),
        "research_word_count": sum(len(t.split()) for t in texts.values()),
        "document_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in texts},
        "plan_sha256": hashlib.sha256((root / "configs/vocabulary_lora_research_plan_v1.json").read_bytes()).hexdigest(),
        "review_sha256": hashlib.sha256((report_dir / "review.json").read_bytes()).hexdigest(),
        "swappable_review_sha256": hashlib.sha256((report_dir / "swappable_review.json").read_bytes()).hexdigest(),
        "swappable_costs_sha256": hashlib.sha256((report_dir / "swappable_costs.json").read_bytes()).hexdigest(),
        "changed_since_review_snapshot": changed_since_snapshot,
        "runtime_tests_run": False, "model_training_run": False,
        "checks": checks,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "checks_passed", "failures", "local_links_checked", "public_source_hashes_checked", "tokenizer_probe_rerun", "research_word_count", "changed_since_review_snapshot")}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

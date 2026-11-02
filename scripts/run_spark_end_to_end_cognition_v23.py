#!/usr/bin/env python3
"""Review or execute one bound Phase 3 V23 End-to-End Semi-Cognition Core, Closed-Loop Multi-Step Controller (k=0 -> k=1), Live-Trajectory Calibrated Monitor Heads, C12 CompiledProcedure, Content-Keyed GPU Titans (S2b), Untagged Salience-Pooled Paraphrase P1/HOPE (S3/S5), Testing-Usage Replica Identity ("replica_test_archivist_v23"), Live Spark E1/E2/E3 Logit Dissociation, Autoregressive Multi-Token Decoding, and Unified Persistent Checkpoint Round-Trip Reload run on MPS."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("SPARK_V23_WORKSPACE", CODE_ROOT)).resolve()
sys.path.insert(0, str(CODE_ROOT / "src"))
sys.path.insert(0, str(CODE_ROOT / "scripts"))

from launch_spark_intrinsic_readout_environment_v1 import validate_environment
from replica_lab.capacity_guard import CapacityGuard, CapacityPolicy
from replica_lab.execution_manifest import build_execution_manifest
from replica_lab.intrinsic_memory_alignment_v19 import validate_alignment_pairs_v19
from replica_lab.intrinsic_memory_budget_v12 import (
    audit_io,
    sha_json,
    validate_plan,
)
from replica_lab.intrinsic_memory_budget_v13 import IncrementalWorkLedger
from replica_lab.intrinsic_memory_counterfactual_v13 import (
    DiagnosticEngineV13,
    restore_parent_checkpoint,
)
from replica_lab.intrinsic_memory_coverage_v12 import (
    SOURCE_FEATURE_BINDING_KEYS,
    SOURCE_FILE_BYTES,
    declared_tokenizer_identity,
    tokenizer_identity_digest,
)
from replica_lab.intrinsic_memory_execution_v5 import bound_json, contained, hash_file
from replica_lab.intrinsic_memory_execution_v12 import (
    record_attempt,
    validate_environment_closure,
)
from replica_lab.intrinsic_memory_native_v5 import base_hash
from replica_lab.intrinsic_memory_native_v12 import close_stage, load_native, reserved_read
from replica_lab.spark_end_to_end_cognition_v23 import (
    END_TO_END_COGNITION_PROTOCOL_V23,
    run_spark_end_to_end_cognition_v23_pipeline,
)
from replica_lab.wall_deadline_v1 import WallPolicy, run_wall_bounded
from run_spark_alignment_v19 import (
    _upgrade_to_independent_reader,
    charged_write,
)
import run_spark_intrinsic_coverage_v12 as native_host

native_host.ROOT = ROOT
from run_spark_intrinsic_coverage_v12 import artifact_bytes, exclusive_worker, host_reading

PACKET_FORMAT = "replica.intrinsic_memory.end_to_end_cognition_launch_packet.v23"
AUTHORIZATION_FORMAT = "replica.intrinsic_memory.end_to_end_cognition_authorization.v23"
ENV_PACKET = "REPLICA_END_TO_END_COGNITION_V23_PACKET"
ENV_DEADLINE = "REPLICA_END_TO_END_COGNITION_V23_DEADLINE"


def environment_only(packet_path: Path) -> dict:
    packet = json.loads(Path(packet_path).read_text())
    protocol = bound_json(ROOT, packet["protocol"])
    return validate_environment(bound_json(ROOT, protocol["environment"]), root=ROOT)


def validate_v23_bindings(
    root: Path, packet_path: Path, *, check_source: bool = False
) -> tuple[dict, dict, dict]:
    packet = json.loads(Path(packet_path).read_text())
    if packet.get("format") != PACKET_FORMAT:
        raise ValueError("unexpected V23 packet format")
    if (
        packet.get("launch_authorized") is not False
        or packet.get("attempts") != 1
        or packet.get("retries") != 0
        or packet.get("native_workers") != 1
        or packet.get("cpu_fallback") is not False
    ):
        raise ValueError("invalid V23 launch packet envelope")

    protocol = bound_json(root, packet["protocol"])
    if protocol.get("format") != END_TO_END_COGNITION_PROTOCOL_V23:
        raise ValueError("unexpected V23 protocol format")
    gen = protocol.get("generation")
    if not isinstance(gen, dict) or gen.get("context_tokens") != 256:
        raise ValueError(
            "V23 protocol missing valid 'generation' contract (context_tokens=256)"
        )
    validate_plan(protocol["budget"])
    validate_plan(protocol["parent_budget"])

    fit_payload = bound_json(root, protocol["data"]["fit"])
    diag_payload = bound_json(root, protocol["data"]["diagnostic_fit"])
    fit_rows = validate_alignment_pairs_v19(fit_payload["rows"])
    diag_rows = validate_alignment_pairs_v19(diag_payload["rows"])
    if {r["world_id"] for r in fit_rows} & {r["world_id"] for r in diag_rows}:
        raise ValueError("fit and diagnostic_fit world IDs must be disjoint")

    stage2_pred = protocol["stage2b_predecessor_outcome"]
    stage2_res_path = contained(root, stage2_pred["result_path"])
    if hash_file(stage2_res_path) != stage2_pred["result_sha256"]:
        raise ValueError("predecessor Stage-2B result digest mismatch")

    stage6_pred = protocol["stage6_v21_predecessor_outcome"]
    stage6_res_path = contained(root, stage6_pred["result_path"])
    if hash_file(stage6_res_path) != stage6_pred["result_sha256"]:
        raise ValueError("predecessor Stage-6 V21 result digest mismatch")

    phase2_pred = protocol["phase2_v22_predecessor_outcome"]
    phase2_res_path = contained(root, phase2_pred["result_path"])
    if hash_file(phase2_res_path) != phase2_pred["result_sha256"]:
        raise ValueError("predecessor Phase-2 V22 result digest mismatch")

    ckpt = protocol["checkpoint"]
    manifest_path = contained(root, str(Path(ckpt["path"]) / "manifest.json"))
    if hash_file(manifest_path) != ckpt["manifest_sha256"]:
        raise ValueError("parent V16 checkpoint manifest digest mismatch")
    ckpt_manifest = json.loads(manifest_path.read_text())
    if ckpt_manifest.get("execution_identity") != ckpt["execution_identity"]:
        raise ValueError("parent V16 checkpoint execution identity mismatch")
    comp_manifest = bound_json(root, protocol["checkpoint_components"])
    if (
        ckpt_manifest.get("components_manifest_sha256")
        != protocol["checkpoint_components"]["sha256"]
    ):
        raise ValueError("parent V16 checkpoint component digest mismatch")
    if comp_manifest.get("adapter_contract") != ckpt_manifest.get("adapter_contract"):
        raise ValueError("parent V16 checkpoint component adapter_contract mismatch")

    source = bound_json(root, protocol["source"])
    source_root = contained(root, source["source_root"])
    if (
        source.get("source_identity") != "official_bf16"
        or source.get("repo_id") != protocol["source_repo"]
        or source.get("revision") != protocol["source_revision"]
        or source.get("derivative") is not False
    ):
        raise ValueError("official BF16 source lineage differs")
    source_execution = bound_json(root, protocol["source_execution"])
    if (
        build_execution_manifest(
            source_root, artifact_sha256=sha_json(source["files"])
        )
        != source_execution
    ):
        raise ValueError("actual model/configuration/local execution code differs")
    declared = protocol.get("source_feature_binding", {})
    for name in SOURCE_FEATURE_BINDING_KEYS:
        if name not in declared:
            raise ValueError("source-feature binding is incomplete")
    if (
        sum(item["bytes"] for item in source["files"]) != SOURCE_FILE_BYTES
        or source.get("total_bytes") != SOURCE_FILE_BYTES
    ):
        raise ValueError("declared source byte size differs from verified receipt")
    if declared["source_files_sha256"] != sha_json(source["files"]):
        raise ValueError("declared source file digest differs")
    expected_identity = declared_tokenizer_identity(source["files"])
    if declared["tokenizer_identity"] != expected_identity:
        raise ValueError("declared tokenizer identity differs from verified source")
    if declared["tokenizer_sha256"] != tokenizer_identity_digest(expected_identity):
        raise ValueError("declared tokenizer digest differs from identity mapping")

    closure_report = validate_environment_closure(root, protocol)
    source_bytes = 0
    if check_source:
        for item in source["files"]:
            path = contained(root, item["path"])
            if path.stat().st_size != item["bytes"] or hash_file(path) != item["sha256"]:
                raise ValueError("source file content differs: " + item["path"])
            source_bytes += item["bytes"]

    output_path = contained(root, packet["output_root"])
    receipt = {
        "format": "replica.intrinsic_memory.end_to_end_cognition_binding_receipt.v23",
        "packet_sha256": hash_file(packet_path),
        "protocol_sha256": packet["protocol"]["sha256"],
        "arm": packet.get("arm", "full_attention_nine"),
        "reader_regime": packet.get("reader_regime", "independent"),
        "predecessor_stage2b_result_sha256": stage2_pred["result_sha256"],
        "predecessor_stage6_v21_result_sha256": stage6_pred["result_sha256"],
        "predecessor_phase2_v22_result_sha256": phase2_pred["result_sha256"],
        "checkpoint_manifest_sha256": ckpt["manifest_sha256"],
        "fit_data_sha256": protocol["data"]["fit"]["sha256"],
        "diagnostic_fit_data_sha256": protocol["data"]["diagnostic_fit"]["sha256"],
        "environment_closure": closure_report,
        "source_bytes_hashed": source_bytes,
        "output_root": packet["output_root"],
        "output_root_unused": not output_path.exists(),
    }
    return packet, protocol, receipt


def validate_v23_authorization(
    root: Path,
    packet_path: Path,
    authorization_path: Path,
    *,
    allow_existing: bool = False,
) -> dict:
    packet, protocol, _ = validate_v23_bindings(root, packet_path)
    auth = json.loads(Path(authorization_path).read_text())
    if auth.get("format") != AUTHORIZATION_FORMAT:
        raise ValueError("unexpected V23 authorization format")
    if auth.get("decision") != "authorize_one_end_to_end_cognition_v23_diagnostic":
        raise ValueError("V23 authorization decision is not active")
    if (
        auth.get("packet_sha256") != hash_file(packet_path)
        or auth.get("protocol_sha256") != packet["protocol"]["sha256"]
        or auth.get("checkpoint_manifest_sha256")
        != protocol["checkpoint"]["manifest_sha256"]
        or auth.get("fit_data_sha256") != protocol["data"]["fit"]["sha256"]
        or auth.get("diagnostic_fit_data_sha256")
        != protocol["data"]["diagnostic_fit"]["sha256"]
        or auth.get("output_root") != packet["output_root"]
        or auth.get("attempts") != 1
        or auth.get("retries") != 0
        or not isinstance(auth.get("human_approval_reference"), str)
        or not auth["human_approval_reference"].strip()
        or not isinstance(auth.get("human_approval_text"), str)
        or not auth["human_approval_text"].strip()
    ):
        raise ValueError("V23 authorization receipt binding mismatch")
    output = contained(root, packet["output_root"])
    if not allow_existing and output.exists():
        raise ValueError("V23 output_root already exists; refusing retry")
    return auth


def worker(args: argparse.Namespace) -> int:
    import psutil
    import torch

    packet = json.loads(args.packet.read_text())
    output = contained(ROOT, packet["output_root"])
    if os.environ.get(ENV_PACKET) != hash_file(args.packet):
        raise ValueError("V23 end-to-end cognition supervision missing")
    validate_v23_authorization(
        ROOT, args.packet, args.authorization, allow_existing=True
    )
    protocol = bound_json(ROOT, packet["protocol"])
    budget = protocol["budget"]
    deadline = float(os.environ[ENV_DEADLINE])
    ledger = IncrementalWorkLedger.create(
        output / "work_ledger.jsonl",
        identity=hash_file(args.packet),
        plan=budget,
        deadline=deadline,
    )
    guard = CapacityGuard(CapacityPolicy(), host_reading())
    if guard.stop_reason:
        raise RuntimeError(guard.stop_reason)
    path = None
    current_stage = "admission"
    try:
        ledger.enter("admission")
        with ledger.work(
            "admission",
            "environment_source_checkpoint_bindings",
            artifact_read_bytes=budget["admission"]["artifact_read_bytes"],
        ) as event:
            environment = environment_only(args.packet)
            _, protocol, receipt = validate_v23_bindings(
                ROOT, args.packet, check_source=True
            )
            observed = (
                receipt["source_bytes_hashed"]
                + receipt["environment_closure"]["bytes_verified"]
            )
            checkpoint = protocol["checkpoint"]
            checkpoint_root = contained(ROOT, checkpoint["path"])
            observed += sum(
                (checkpoint_root / "components" / name).stat().st_size
                for name in ("extension.safetensors", "optimizer.pt")
            )
            observed += (ROOT / protocol["data"]["fit"]["path"]).stat().st_size
            observed += (
                ROOT / protocol["data"]["diagnostic_fit"]["path"]
            ).stat().st_size
            event["actual"] = {"artifact_read_bytes": observed}
            event["source_bytes_hashed"] = receipt["source_bytes_hashed"]
            event["environment_closure"] = receipt["environment_closure"]
        charged_write(
            ledger,
            output,
            "preflight",
            {
                **receipt,
                "environment_validation": environment,
                "native_execution": False,
                "development_labels_parsed": False,
            },
            stage="admission",
        )
        ledger.leave("admission", status="completed")

        current_stage = "load"
        ledger.enter("load")
        arm_name = protocol.get("arm", "full_attention_nine")
        load_protocol = {
            **protocol,
            "intrinsic_placement": "block_input_query_to_post_block_residual",
            "training": {
                **protocol["training"],
                "lr": float(
                    protocol["training"].get("lr", protocol["training"]["wake_lr"])
                ),
            },
        }
        with ledger.work(
            "load",
            "native_code_cache_artifacts",
            writes=16,
            artifact_bytes=1024**2,
        ):
            path, backend, spec, initial_hash, stage_ref = load_native(
                ROOT, load_protocol, "two", ledger, "load"
            )
        checkpoint = protocol["checkpoint"]
        checkpoint_root = contained(ROOT, checkpoint["path"])
        checkpoint_bytes = sum(
            p.stat().st_size for p in checkpoint_root.rglob("*") if p.is_file()
        )
        with ledger.work(
            "load",
            "trained_extension_optimizer_restore",
            restart_restores=1,
            artifact_read_bytes=checkpoint_bytes,
        ) as event:
            migration = restore_parent_checkpoint(
                path, checkpoint_root, checkpoint["execution_identity"]
            )
            event["actual"] = {"artifact_read_bytes": checkpoint_bytes}
            event["checkpoint_manifest_sha256"] = checkpoint["manifest_sha256"]
            event["parent_execution_migration"] = migration
        if arm_name != "two":
            _upgrade_to_independent_reader(
                path,
                arm_name,
                protocol["intrinsic_placement"],
                float(protocol["training"]["wake_lr"]),
            )
        for module in path.modules().values():
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        ledger.leave("load", status="completed")

        current_stage = "diagnose"
        ledger.enter("diagnose")
        engine = DiagnosticEngineV13(
            path,
            backend,
            spec,
            ledger,
            max_new_tokens=int(protocol.get("generation", {}).get("max_new_tokens", 8)),
            output=output,
            stage_ref=stage_ref,
        )
        engine.stage = "diagnose"
        fit_rows = json.loads(
            reserved_read(
                ledger,
                "diagnose",
                "fit_split_read",
                ROOT / protocol["data"]["fit"]["path"],
            )
        )["rows"]
        diag_rows = json.loads(
            reserved_read(
                ledger,
                "diagnose",
                "diagnostic_fit_split_read",
                ROOT / protocol["data"]["diagnostic_fit"]["path"],
            )
        )["rows"]

        last_driver = 0.0
        last_host = 0.0
        with (
            (output / "driver_samples.jsonl").open("x") as driver_stream,
            (output / "host_samples.jsonl").open("x") as host_stream,
        ):

            def observe():
                nonlocal last_driver, last_host
                now = time.monotonic()
                if now - last_driver >= 0.25:
                    sample = {
                        "monotonic": now,
                        "pid": os.getpid(),
                        "rss_bytes": psutil.Process().memory_info().rss,
                        "mps_driver_bytes": int(
                            torch.mps.driver_allocated_memory()
                        ),
                        "mps_active_bytes": int(
                            torch.mps.current_allocated_memory()
                        ),
                    }
                    guard.observe_profile(sample)
                    raw = json.dumps(sample) + "\n"
                    with engine.measured(
                        "driver_sample",
                        writes=1,
                        artifact_bytes=len(raw.encode()),
                    ):
                        driver_stream.write(raw)
                        driver_stream.flush()
                    last_driver = now
                if now - last_host >= 2.0:
                    reading = host_reading()
                    guard.observe_host(reading)
                    raw = json.dumps({"monotonic": now, **reading}) + "\n"
                    with engine.measured(
                        "host_sample",
                        writes=1,
                        artifact_bytes=len(raw.encode()),
                    ):
                        host_stream.write(raw)
                        host_stream.flush()
                    last_host = now
                snap = ledger.snapshot(cached=True)
                deadlines = ledger.deadline_state(snap)
                if deadlines["outer_remaining_seconds"] <= 0 or any(
                    v <= 0
                    for v in deadlines["stage_remaining_seconds"].values()
                ):
                    raise RuntimeError("stage or outer V23 end-to-end cognition deadline")
                if artifact_bytes(output) > sum(
                    st["artifact_bytes"] for st in budget.values()
                ):
                    raise RuntimeError("unreserved V23 end-to-end cognition artifact growth")
                if guard.stop_reason:
                    raise RuntimeError(guard.stop_reason)

            observe()
            pipeline_report = run_spark_end_to_end_cognition_v23_pipeline(
                engine,
                fit_rows,
                diagnostic_rows=diag_rows,
                wake_steps=int(protocol["training"]["wake_steps"]),
                sleep_steps=int(protocol["training"]["sleep_steps"]),
                sleep_lr=float(protocol["training"]["sleep_lr"]),
                core_steps=int(protocol["training"].get("core_steps", 36)),
                head_steps=int(protocol["training"].get("head_steps", 75)),
                top_k=int(protocol["training"]["top_k"]),
                require_competence=True,
                observe=observe,
                seed=int(protocol["training"].get("seed", 17)),
            )
            charged_write(
                ledger,
                output,
                "end_to_end_cognition_v23_pipeline",
                pipeline_report,
                stage="diagnose",
            )
        close_stage(engine, "diagnose", "completed")

        current_stage = "finalize"
        ledger.enter("finalize")
        unchanged = base_hash(path.adapter.base) == initial_hash
        if not unchanged:
            raise ValueError(
                "original parameter bytes changed during V23 end-to-end cognition diagnostic"
            )
        snap = ledger.snapshot()
        charged_write(
            ledger,
            output,
            "result",
            {
                "format": "replica.intrinsic_memory.end_to_end_cognition_result.v23",
                "packet_sha256": hash_file(args.packet),
                "protocol_sha256": packet["protocol"]["sha256"],
                "checkpoint_manifest_sha256": protocol["checkpoint"][
                    "manifest_sha256"
                ],
                "v21_continuum": pipeline_report["v21_continuum"],
                "v23a_end_to_end_cognitive_core": pipeline_report[
                    "v23a_end_to_end_cognitive_core"
                ],
                "v23b_live_trajectory_heads_and_compilation": pipeline_report[
                    "v23b_live_trajectory_heads_and_compilation"
                ],
                "v23c_content_keyed_titans_and_untagged_paraphrase": pipeline_report[
                    "v23c_content_keyed_titans_and_untagged_paraphrase"
                ],
                "v23d_replica_identity_dissociation_decoding_and_rs4": pipeline_report[
                    "v23d_replica_identity_dissociation_decoding_and_rs4"
                ],
                "complete_restoration_verified": pipeline_report[
                    "complete_restoration_verified"
                ],
                "all_v23_gates_passed": pipeline_report["all_v23_gates_passed"],
                "original_weights_unchanged": unchanged,
                "capacity": guard.report(),
                "artifact_bytes_observed": artifact_bytes(output),
                "reservations": snap["reserved"],
                "audit_io": audit_io(ledger),
                "deadline_state": ledger.deadline_state(snap),
                "development_labels_used": False,
                "sealed_evaluation_labels_opened": False,
                "pre_attempt_admission": str(
                    output.with_name(output.name + "_preflight")
                ),
            },
            stage="finalize",
        )
        ledger.leave("finalize", status="completed")
        return 0
    except BaseException as error:
        charged_write(
            ledger,
            output,
            "failure",
            {
                "type": type(error).__name__,
                "message": str(error),
                "reserved_work_not_refunded": True,
                "development_labels_used": False,
            },
            stage=current_stage,
        )
        raise
    finally:
        if path is not None:
            path.close()
        try:
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
                torch.mps.empty_cache()
            children = psutil.Process(os.getpid()).children(recursive=True)
            if children:
                psutil.wait_procs(children, timeout=2.0)
        except Exception:
            pass


def launch(args: argparse.Namespace) -> int:
    packet = json.loads(args.packet.read_text())
    protocol = bound_json(ROOT, packet["protocol"])
    validate_v23_authorization(ROOT, args.packet, args.authorization)
    budget = protocol["budget"]
    parent_budget = protocol["parent_budget"]
    wall_seconds = int(protocol["wall_seconds"])
    output = contained(ROOT, packet["output_root"])
    marker = contained(ROOT, packet["attempt_marker"])
    if marker.resolve().is_relative_to(output.resolve()):
        raise ValueError("attempt marker must be outside the output tree")
    with exclusive_worker():
        preflight = output.with_name(output.name + "_preflight")
        preflight.mkdir(parents=True, exist_ok=False)
        admission = IncrementalWorkLedger.create(
            preflight / "ledger.jsonl",
            identity=hash_file(args.packet),
            plan=parent_budget,
            deadline=time.monotonic() + wall_seconds + 180,
        )
        admission.enter("admission")
        with admission.work(
            "admission",
            "validate_before_attempt_or_model_load",
            artifact_read_bytes=budget["admission"]["artifact_read_bytes"],
        ):
            _, protocol, checked = validate_v23_bindings(
                ROOT, args.packet, check_source=True
            )
        charged_write(admission, preflight, "receipt", checked, stage="admission")
        admission.leave("admission", status="completed")
        admission.enter("supervise")
        with admission.work(
            "supervise",
            "attempt_and_supervisor_artifacts",
            writes=32,
            artifact_bytes=4 * 1024**2,
            artifact_read_bytes=16 * 1024**2,
        ) as parent_event:
            record_attempt(
                marker,
                packet_sha256=hash_file(args.packet),
                protocol_sha256=packet["protocol"]["sha256"],
                arm=packet.get("arm", "full_attention_nine"),
                output_root=packet["output_root"],
                authorization_sha256=hash_file(args.authorization),
            )
            output.mkdir(parents=True, exist_ok=False)
            env = {
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": str(ROOT / "src"),
                "SPARK_V23_WORKSPACE": str(ROOT),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTORCH_ENABLE_MPS_FALLBACK": "0",
                "HF_MODULES_CACHE": str(output / "hf_modules"),
                ENV_PACKET: hash_file(args.packet),
                ENV_DEADLINE: str(time.monotonic() + wall_seconds - 11),
            }
            command = [
                sys.executable,
                str(ROOT / "scripts/run_spark_end_to_end_cognition_v23.py"),
                "--packet",
                str(args.packet),
                "--authorization",
                str(args.authorization),
                "--worker",
            ]
            result = run_wall_bounded(
                command,
                cwd=ROOT,
                output=output / "supervision",
                policy=WallPolicy(wall_seconds),
                launch_authorized=True,
                authorization_identity=hash_file(args.authorization),
                env=env,
            )
            parent_event["actual"] = {
                "artifact_bytes": artifact_bytes(output / "supervision")
                + marker.stat().st_size,
                "writes": 1
                + sum(
                    p.is_file() for p in (output / "supervision").rglob("*")
                ),
            }
        admission.leave("supervise", status="completed")
        return result.exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--authorization", type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--environment-only", action="store_true")
    modes.add_argument("--review", action="store_true")
    modes.add_argument("--launch", action="store_true")
    modes.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    args.packet = args.packet.resolve()
    if args.environment_only:
        print(json.dumps(environment_only(args.packet), indent=2, sort_keys=True))
        return 0
    if args.review:
        environment = environment_only(args.packet)
        _, _, receipt = validate_v23_bindings(ROOT, args.packet)
        print(
            json.dumps(
                {
                    "environment_validation": environment,
                    **receipt,
                    "launch_authorized": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.authorization is None:
        parser.error("an exact V23 authorization receipt is required")
    args.authorization = args.authorization.resolve()
    if args.launch:
        return launch(args)
    return worker(args)


if __name__ == "__main__":
    raise SystemExit(main())

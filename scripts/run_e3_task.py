#!/usr/bin/env python3
"""Run bounded E3 task decisions on MPS against one fixture case.

The exact journal persists between invocations; a fresh process resumes from it
without re-executing completed actions. The sealed source is loaded read-only;
only extension tensors are applied. Raw prompts, completions, receipts, state
digests and budgets are retained in the session report. Generated text is never
admitted as evidence, and the host-only reference repair is never exposed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1", "PYTORCH_ENABLE_MPS_FALLBACK": "0",
    "TOKENIZERS_PARALLELISM": "false",
}
MODES = ("full", "core_disabled", "workspace_zero", "workspace_shuffled", "ff_core",
         "extra_inference")
IMPLEMENTATION_FILES = (
    "scripts/run_e3_task.py",
    "src/replica_lab/runtime/task_graph.py",
    "src/replica_lab/runtime/journal.py",
    "src/replica_lab/runtime/actions.py",
    "src/replica_lab/runtime/tools.py",
    "src/replica_lab/runtime/workspace_store.py",
    "src/replica_lab/runtime/loop.py",
    "src/replica_lab/architecture/contracts.py",
)
MAX_DECISION_LIMIT = 64
DEFAULT_MAX_PROMPT_TOKENS = 2048
DEFAULT_GENERATED_TOKEN_BUDGET = 128
DEFAULT_MAX_TOOL_CALLS = 64
SEED_POLICY_TEXT = ("The evaluator tests are authoritative; only files under workspace/ "
                    "may be edited.")


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_budget(args):
    """Cumulative exact-state budget; token totals may be overridden explicitly."""
    decisions = int(args.max_decisions)
    tool_calls = int(args.max_tool_calls)
    prompt_tokens = (int(args.max_total_prompt_tokens)
                     if args.max_total_prompt_tokens is not None
                     else decisions * int(args.max_prompt_tokens))
    generated_tokens = (int(args.max_total_generated_tokens)
                        if args.max_total_generated_tokens is not None
                        else decisions * int(args.max_new_tokens))
    return {"decisions": decisions, "tool_calls": tool_calls,
            "prompt_tokens": prompt_tokens, "generated_tokens": generated_tokens}


def build_core_spec(args):
    from replica_lab.architecture.contracts import CoreSpec, PortSpec
    spec = CoreSpec(width=int(args.core_width), slots=int(args.core_slots),
                    steps=int(args.core_steps), heads=int(args.core_heads),
                    task_features=int(args.core_task_features),
                    ffn_hidden=int(args.core_ffn_hidden),
                    control_hidden=int(args.core_control_hidden))
    spec.validate(PortSpec())
    return spec


def seed_policy_record(scope):
    """One deterministic host policy record; generated text is never admitted."""
    from replica_lab.memory_system import MemoryRecord
    digest = hashlib.sha256(json.dumps(
        {"text": SEED_POLICY_TEXT, "subject": scope.subject_id,
         "branch": scope.branch_id, "purpose": scope.purpose},
        sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return MemoryRecord(
        record_id=f"policy-{digest[:24]}", subject_id=scope.subject_id,
        branch_id=scope.branch_id, text=SEED_POLICY_TEXT, observed_at=0, event_at=0,
        purpose=scope.purpose, kind="policy", source_id="host-seed-policy",
        authenticity="verified_source")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("mps",), default="mps")
    parser.add_argument("--case", type=Path, required=True,
                        help="Fixture case directory containing case.json")
    parser.add_argument("--journal", type=Path, required=True,
                        help="Persistent exact-task journal (SQLite)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Fresh evidence/report directory")
    parser.add_argument("--model-path", type=Path,
                        default=ROOT / "checkpoints/minicpm5_source")
    parser.add_argument("--model-lock", type=Path,
                        default=ROOT / "configs/minicpm5_sealed_lock.local.json")
    parser.add_argument("--extension", type=Path,
                        help="Compatible extension-only export directory")
    parser.add_argument("--accept-parent-v1", action="store_true",
                        help="Migrate the E2 v1 parent extension per the E3 loading contract")
    parser.add_argument("--mode", choices=MODES, default="full")
    parser.add_argument("--max-decisions", type=int, default=4,
                        help="Cumulative exact-state decision cap")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Decisions in this process (default: --max-decisions); "
                             "the cumulative cap is preserved across restart")
    parser.add_argument("--max-tool-calls", type=int, default=DEFAULT_MAX_TOOL_CALLS,
                        help="Cumulative exact-state tool-call budget")
    parser.add_argument("--max-new-tokens", "--max-generated-tokens", dest="max_new_tokens",
                        type=int, default=DEFAULT_GENERATED_TOKEN_BUDGET,
                        help="Per-decision generated-token cap")
    parser.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS,
                        help="Per-decision prompt-token cap")
    parser.add_argument("--max-total-prompt-tokens", type=int, default=None,
                        help="Cumulative prompt-token cap (default max-decisions * per-decision)")
    parser.add_argument("--max-total-generated-tokens", type=int, default=None,
                        help="Cumulative generated-token cap (default derived)")
    parser.add_argument("--core-width", type=int, default=256)
    parser.add_argument("--core-slots", type=int, default=8)
    parser.add_argument("--core-steps", type=int, default=2)
    parser.add_argument("--core-heads", type=int, default=4)
    parser.add_argument("--core-task-features", type=int, default=24)
    parser.add_argument("--core-ffn-hidden", type=int, default=1024)
    parser.add_argument("--core-control-hidden", type=int, default=340)
    parser.add_argument("--subject", default=None, help="Defaults to the case id")
    parser.add_argument("--branch", default="e3-task")
    parser.add_argument("--purpose", default="e3-connected-task")
    parser.add_argument("--ledger", type=Path,
                        help="Persistent record ledger; default beside the journal")
    parser.add_argument("--workspace-dir", type=Path,
                        help="Persistent workspace store; default beside the journal")
    parser.add_argument("--shuffled-workspace-dir", type=Path,
                        help="Another task's workspace store; required for workspace_shuffled")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--seed-policy", action=argparse.BooleanOptionalAction, default=True,
                        help="Admit one deterministic host policy record for the case scope")
    return parser


def validate_arguments(args):
    if not 1 <= args.max_decisions <= MAX_DECISION_LIMIT:
        raise ValueError(f"--max-decisions must be 1..{MAX_DECISION_LIMIT}")
    if args.max_steps is not None and not 1 <= args.max_steps <= args.max_decisions:
        raise ValueError("--max-steps must be 1..--max-decisions")
    if not 1 <= args.max_new_tokens <= 512 or not 256 <= args.max_prompt_tokens <= 4096:
        raise ValueError("Token budgets must be 1..512 generated and 256..4096 prompt tokens")
    if not 0 <= args.max_tool_calls <= 4096:
        raise ValueError("--max-tool-calls must be 0..4096")
    budget = resolve_budget(args)
    if budget["prompt_tokens"] < 0 or budget["generated_tokens"] < 0:
        raise ValueError("Cumulative token budgets must be nonnegative")
    if min(args.core_width, args.core_slots, args.core_steps, args.core_heads,
           args.core_task_features, args.core_ffn_hidden,
           args.core_control_hidden) <= 0:
        raise ValueError("Core spec values must be positive integers")
    if args.core_steps > 8:
        raise ValueError("--core-steps is bounded to 8 tied iterations")
    if args.core_width % args.core_heads:
        raise ValueError("--core-width must divide --core-heads")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise ValueError("Use a fresh empty output directory; prior evidence is preserved")
    if args.mode == "workspace_shuffled" and args.shuffled_workspace_dir is None:
        raise ValueError("--mode workspace_shuffled requires --shuffled-workspace-dir")
    model = args.model_path.resolve()
    for name in ("output", "journal", "ledger", "workspace_dir"):
        value = getattr(args, name)
        if value is not None and value.resolve().is_relative_to(model):
            raise ValueError(f"{name} must not modify the sealed original model directory")
    if str(args.journal) == ":memory:":
        raise ValueError("The runtime entry point requires a persistent journal path")


def read_case(case_root):
    from replica_lab.runtime.tools import FixtureWorkspace
    tools = FixtureWorkspace(case_root)
    case = {
        "case_id": tools.case_id, "role": tools.role, "goal": tools.goal,
        "success_condition": dict(tools.success_condition),
        "constraints": list(tools.constraints), "allowed_tools": list(tools.allowed_tools),
        "evaluator": tools.evaluator_rel,
    }
    return tools, case


def component_versions():
    return {"runtime": "e3-runtime-v1", "tools": "fixture-workspace-v1",
            "journal": "exact-journal-v1", "task_features": "e3-layout-v1"}


def build_initial_state(tools, args):
    from replica_lab.runtime.task_graph import (Budget, Constraint, ExactTaskState,
                                                SuccessCondition)
    declared = resolve_budget(args)
    budget = Budget(declared["decisions"], declared["tool_calls"],
                    declared["prompt_tokens"], declared["generated_tokens"])
    return ExactTaskState(
        task_id=tools.case_id, subject=args.subject or tools.case_id, branch=args.branch,
        purpose=args.purpose, source_epoch=0, goal=tools.goal,
        success_conditions=(SuccessCondition(tools.success_condition["condition_id"],
                                             tools.success_condition["description"]),),
        constraints=tuple(Constraint(f"con-{index}", text)
                          for index, text in enumerate(tools.constraints)),
        subgoals=(), claims=(), evidence=(), obligations=(), actions=(), outcomes=(),
        notes=(), budget=budget,
        binding={**component_versions(),
                 "case_format": "replica.e3.fixture.v1"},
        allowed_tools=tuple(tools.allowed_tools), revision=0)


def journal_binding(tools, args):
    from replica_lab.architecture.contracts import fingerprint
    versions = component_versions()
    versions["budget"] = fingerprint(resolve_budget(args))
    return {
        "task_id": tools.case_id, "subject": args.subject or tools.case_id,
        "branch": args.branch, "purpose": args.purpose, "source_epoch": 0,
        "goal_digest": fingerprint(tools.goal),
        "component_versions": versions,
    }


def proposer_for_mode(session, tokenizer, tools, args, scope):
    from replica_lab.runtime.loop import SessionProposer
    common = dict(task_id=tools.case_id, goal=tools.goal, scope=scope, device=args.device)
    if args.mode == "core_disabled":
        return SessionProposer(session, tokenizer, enabled_branches=("memory",),
                               core_disabled=True, **common)
    if args.mode == "workspace_zero":
        return SessionProposer(session, tokenizer, zero_previous_workspace=True, **common)
    if args.mode == "ff_core":
        return SessionProposer(session, tokenizer, core_impl="feedforward", **common)
    if args.mode == "extra_inference":
        return SessionProposer(session, tokenizer, enabled_branches=("memory",),
                               extra_inference_pass=True, **common)
    return SessionProposer(session, tokenizer, **common)


def build_retriever(session, scope):
    """Host-owned retrieval observations: admitted record identities only."""

    def retrieve(query):
        from dataclasses import asdict
        selection = session.index.retrieve(query, **asdict(scope), limit=4)
        if not selection.records:
            return "(no eligible records)"
        return "; ".join(f"{record.record_id} {record.record_hash}"
                         for record in selection.records)

    return retrieve


def execute(args, tools, case, report):
    import torch
    from replica_lab.architecture.loading import load_local_mps
    from replica_lab.memory_system import EligibilityScope, RecordLedger
    from replica_lab.memory_system.session import MemorySession
    from replica_lab.runtime.journal import TaskJournal
    from replica_lab.runtime.loop import TaskRuntime
    from replica_lab.runtime.workspace_store import WorkspaceStore

    torch.manual_seed(args.seed)
    core_spec = build_core_spec(args)
    adapter, tokenizer, lock = load_local_mps(args.model_path, args.model_lock,
                                             core=core_spec)
    if adapter.manifest.core != core_spec:
        raise ValueError("Loaded manifest does not carry the declared core spec")
    report["source_receipt"] = {
        "path": str(args.model_path.resolve()), "lock_path": str(args.model_lock.resolve()),
        "lock_sha256": file_digest(args.model_lock), "repo_id": lock["repo_id"],
        "revision": lock["revision"], "seal_validated": True,
    }
    report["core_spec"] = asdict(core_spec)
    report["manifest"] = adapter.manifest.to_dict()
    report["manifest_digest"] = adapter.manifest.digest
    if args.extension:
        if args.accept_parent_v1:
            receipt = adapter.load_extension(args.extension, accept_parent_v1=True)
        else:
            receipt = adapter.load_extension(args.extension)
        if adapter.manifest.core != core_spec:
            raise ValueError("Extension manifest core spec differs from --core-* arguments")
        report["extension"] = {"path": str(args.extension.resolve()), "receipt": receipt}
    else:
        report["extension"] = None
        report["component_status"] = "new_modules_untrained; projections_zero_initialized"

    declared_budget = resolve_budget(args)
    journal = TaskJournal.open(args.journal, journal_binding(tools, args))
    workspace_dir = args.workspace_dir or args.journal.parent / "task_workspace"
    ledger_path = args.ledger or args.journal.parent / "memory_ledger.sqlite"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    workspace_store = WorkspaceStore(workspace_dir)

    if args.mode == "workspace_shuffled":
        from replica_lab.runtime.workspace_store import WorkspaceStore as WS
        shuffled = WS(args.shuffled_workspace_dir).load()
        if shuffled is None:
            raise ValueError("--shuffled-workspace-dir holds no loadable workspace")
    else:
        shuffled = None

    runtime = TaskRuntime(
        tools, journal, workspace_store=workspace_store, mode=args.mode,
        max_prompt_tokens=args.max_prompt_tokens,
        generated_token_budget=args.max_new_tokens, shuffled_workspace=shuffled,
        action_guard=args.action_guard)
    if not journal.initialize_done():
        runtime.initialize(build_initial_state(tools, args))
    recovered = runtime.recover()
    report["journal"] = {"path": str(args.journal.resolve()), "head": journal.head,
                         "epoch": journal.epoch, "binding": journal.binding,
                         "recovered_intents": recovered}

    scope = EligibilityScope(args.subject or tools.case_id, args.branch, args.purpose, 0,
                             "assistant_for_source", None)
    with RecordLedger(ledger_path) as ledger:
        if args.seed_policy:
            policy = seed_policy_record(scope)
            inserted = ledger.admit(policy, authorized_subject=scope.subject_id,
                                    authorized_branch=scope.branch_id)
            report["seed_policy"] = {"record_id": policy.record_id,
                                     "observed_at": policy.observed_at,
                                     "inserted": bool(inserted)}
        else:
            report["seed_policy"] = None
        session = MemorySession(adapter, tokenizer, ledger)
        proposer = proposer_for_mode(session, tokenizer, tools, args, scope)
        runtime.retriever = build_retriever(session, scope)
        remaining = (runtime.state.budget.max_decisions
                     - runtime.state.budget.used_decisions)
        run_limit = args.max_steps if args.max_steps is not None else args.max_decisions
        decision_limit = max(0, min(run_limit, remaining))
        summary = runtime.run(proposer, max_decisions=decision_limit)
        report["final_state"] = runtime.summary()
        report["workspace_store"] = {
            "path": str(Path(workspace_dir).resolve()),
            "loaded": workspace_store.load() is not None,
        }
    report["budgets"] = {
        "declared": declared_budget,
        "exact_state_cumulative": declared_budget,
        "per_decision": {"prompt_tokens": args.max_prompt_tokens,
                         "generated_tokens": args.max_new_tokens,
                         "max_prompt_tokens": args.max_prompt_tokens,
                         "max_new_tokens": args.max_new_tokens,
                         "source": "ProposalRequest / SessionProposer caps"},
        "derived_totals": {
            "prompt_tokens": args.max_total_prompt_tokens is None,
            "generated_tokens": args.max_total_generated_tokens is None,
        },
    }
    report["costs"] = {
        "decisions": summary["decisions"],
        "decision_wall_seconds": [item["decision_wall_seconds"]
                                  for item in summary["trace"]],
        "wall_seconds": sum(item["decision_wall_seconds"] for item in summary["trace"]),
        "prompt_tokens": summary["budget"]["used_prompt_tokens"],
        "generated_tokens": summary["budget"]["used_generated_tokens"],
        "tool_calls": summary["budget"]["used_tool_calls"],
    }
    report["summary"] = summary
    report["status"] = "completed_e3_task"


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_arguments(args)
    except ValueError as error:
        parser.error(str(error))
    # These variables are established before importing any neural package.
    os.environ.update(OFFLINE_ENVIRONMENT)
    args.output.mkdir(parents=True, exist_ok=True)
    tools, case = read_case(args.case)
    report = {
        "schema_version": "replica.session.e3.v1", "status": "started",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "offline_environment": dict(OFFLINE_ENVIRONMENT), "device": "mps",
        "parameter_and_compute_dtype": "float32", "downloads": 0, "optimizer_updates": 0,
        "checkpoints_saved": 0, "seed": args.seed, "case": case,
        "free_disk_before_bytes": shutil.disk_usage(args.output).free,
        "implementation_sha256": {path: file_digest(ROOT / path)
                                  for path in IMPLEMENTATION_FILES},
        "limitations": [
            "Generated text is a proposal, never evidence or a receipt",
            "Success is decided only by the independent host evaluator path",
            "Extension loading does not establish training quality",
        ],
        "notes": ["reference_repair is host-only and never serialized here"],
    }
    profiler = None
    try:
        from replica_lab.profiling import RunProfiler
        from replica_lab.architecture.loading import exclusive_mps_worker
        profiler = RunProfiler(args.device, args.output, sample_interval=0.25).start()
        with exclusive_mps_worker(ROOT):
            execute(args, tools, case, report)
    except BaseException as error:
        report.update(status="failed_e3_task", error=str(error),
                      error_type=type(error).__name__, traceback=traceback.format_exc())
        raise
    finally:
        if profiler is not None:
            report["profile"] = profiler.stop()
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["free_disk_after_bytes"] = shutil.disk_usage(args.output).free
        report.pop("reference_repair", None)
        (args.output / "session_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = report.get("summary", {})
    print(json.dumps({"status": report["status"], "case_id": case["case_id"],
                      "decisions": summary.get("decisions"), "success": summary.get("success"),
                      "report": str(args.output / "session_report.json")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

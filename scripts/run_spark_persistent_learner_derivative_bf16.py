#!/usr/bin/env python3
"""Derivative BF16 P1/spine orchestration runner.

This is the reviewable entrypoint for the FP8-derived BF16 P1/spine experiment
declared by ``configs/spark_persistent_learner_derivative_bf16_protocol_v1.json``.
It reuses the versioned v2 schedule, direct-response-CE/P1 controls, prepared
fixture and checkpoint/restart contracts, but binds the active source to the
dequantized derivative identity instead of the official FP8 source.

Two paths exist:

* ``--backend cpu_smoke --stage smoke`` -- a small CPU synthetic derivative
  fixture that proves the changed path (derivative precision binding, zero-init
  parity, finite P/port gradients, one actual NativeFusedAdamW update with
  moments/clock, frozen base, extension-only reload) and refuses official
  FP8/BF16 substitution.  It is engineering-only and is never native evidence.
* ``--backend native_derivative`` -- the real MPS path.  It refuses before any
  model load unless the derivative manifest/shard hashes, current P1/spine
  admission, implementation hashes, protocol commit and an *external*
  derivative-specific authorization all bind exactly.  No authorization exists
  for this precision identity today and this runner never manufactures one.

No downloads, no sealed-final access, no official FP8/BF16 substitution, no CPU
fallback and no useful-learning claim.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
for _relative in (ROOT / "src", ROOT / "scripts", ROOT / "tests"):
    if str(_relative) not in sys.path:
        sys.path.insert(0, str(_relative))

from preflight_spark_persistent_learner_derivative_bf16 import (  # noqa: E402
    CONFIG_PATH, PROTOCOL_ID, ProtocolError as PreflightError, _missing_bindings,
    _validate_authorization, read_json, sha256_file, validate_config,
)
from replica_lab import artifact_publication as publication  # noqa: E402
from replica_lab.architecture.spark2_5 import (  # noqa: E402
    DEQUANTIZED_FP8_DERIVATIVE, DERIVATIVE_ACTIVATION_SEMANTICS,
    DERIVATIVE_EXECUTION_FORMAT, OFFICIAL_NATIVE_FP8, P1_MEMORY_SIZE,
    P1_PORT_HEADS, P1_SPINE_SITES, P1_SPINE_SLOTS, Spark2_5ContractError,
    classify_precision, dequantized_fp8_derivative, evaluate_derivative_p1_spine,
    load_spark2_5_config, load_tokenizer_identity, official_bf16_precision,
    official_native_fp8_precision, resolve_thinking_mode,
)
from replica_lab.architecture.spark_adapter import (  # noqa: E402
    Spark2_5Adapter, SparkAdapterError,
)
from replica_lab.architecture.loading import exclusive_mps_worker  # noqa: E402
from replica_lab.architecture.spine_memory import SpineMemoryExperiment  # noqa: E402
from replica_lab.architecture.spine_text import (  # noqa: E402
    TEXT_CODEC_VERSION, TEXT_REPRESENTATION_VERSION, SparkTextMemoryPath,
)
from replica_lab.contextual_spark_fixture import (  # noqa: E402
    ContextualSparkCausalLM, contextual_config_raw, ids_and_mask, labels_from_ids,
    open_memory_ports, populate_contextual_evidence,
)
from replica_lab.p1_response_learning import (  # noqa: E402
    build_port_optimizer, direct_response_ce_step, prepare_arm_query,
)
from replica_lab.persistent_learner_runner import (  # noqa: E402
    ARMS, FIT_RECEIPT_FORMAT, PreparedDataset, Refused, WorkLedger,
    _json_safe, _merge_restart_report, _read_json, _read_jsonl, _sha256,
    _write_json, _write_jsonl, _write_storage_receipts, load_fit_state,
    run_development, run_fit, run_restart,
)
from replica_lab.spark_text_memory_runner import Backend  # noqa: E402

RUN_FORMAT = "replica.spark_persistent_learner_derivative.run.v1"
SMOKE_FORMAT = "replica.spark_persistent_learner_derivative.smoke.v1"
EXTENSION_RELOAD_FORMAT = (
    "replica.spark_persistent_learner_derivative.extension_reload.v1")
NATIVE_LABEL = "native_derivative"
SMOKE_LABEL = "cpu_synthetic_derivative_fixture"
CPU_LABEL = "cpu"
STAGES = ("smoke", "all", "fit", "development", "restart")
NATIVE_STAGES = ("all", "fit", "development", "restart")
SMOKE_RECORDS = (
    ("answer", "cobalt alpha is the verified response token"),
    ("near", "cobalt beta is a nearby distractor"),
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require(condition: bool, check: str, detail: str) -> None:
    if not condition:
        raise Refused(check, detail)


# ---------------------------------------------------------------------------
# Derivative protocol/dataset wrappers over the reused v2 contracts


@dataclass(frozen=True)
class DerivativeProtocol:
    """The v2 schedule/config surface bound to the derivative identity."""

    path: Path
    sha256: str
    config: Mapping[str, Any]
    report: Mapping[str, Any] = field(repr=False)

    @property
    def protocol_id(self) -> str:
        return str(self.config["protocol_id"])

    @property
    def root(self) -> Path:
        return self.path.parent.parent

    @property
    def schedule(self) -> Mapping[str, Any]:
        return self.config["work_and_cost"]["native_schedule"]

    @property
    def source_identity(self) -> Mapping[str, Any]:
        return self.config["source_identities"]["dequantized_fp8_derivative"]

    @property
    def source(self) -> Mapping[str, Any]:
        return self.report["source"]

    @property
    def derivative_config(self):
        return self.report["derivative_config"]

    @property
    def precision(self):
        return self.report["precision_identity"]

    @property
    def admission(self) -> Mapping[str, Any]:
        return self.report["admission"]

    @classmethod
    def load(cls, path: str | Path, *, full_hash: bool = True) -> "DerivativeProtocol":
        path = Path(path).resolve()
        _require(path.is_file(), "protocol", f"missing protocol {path}")
        config = read_json(path)
        try:
            report = validate_config(config, path.parent.parent, full_hash=full_hash)
        except PreflightError as error:
            raise Refused(error.check, error.detail) from error
        _require(config["launch"].get("runner_status") == "implemented", "runner",
                 "the derivative runner is not bound as implemented in the protocol")
        _require(config["launch"].get("runner") ==
                 "scripts/run_spark_persistent_learner_derivative_bf16.py", "runner",
                 "the protocol binds a different runner")
        return cls(path=path, sha256=sha256_file(path), config=config, report=report)


class DerivativeDataset(PreparedDataset):
    """The prepared v2 fixture, reopened against the derivative protocol hash.

    The fixture manifest is byte-identical and bound by hash in the derivative
    protocol.  Only the fit-receipt barrier is rebased from the v2 protocol id
    to the derivative protocol hash so a derivative receipt is the only key
    that can open development labels.
    """

    def open_development(self, receipt: Mapping[str, Any]) -> None:
        _require(receipt.get("format") == FIT_RECEIPT_FORMAT, "fit_receipt",
                 "format differs")
        _require(receipt.get("protocol_sha256") == self.protocol_sha256,
                 "fit_receipt", "protocol hash differs")
        _require(receipt.get("prepared_manifest_sha256") == self.manifest_sha256,
                 "fit_receipt", "manifest hash differs")
        _require(receipt.get("status") == "fit_executed"
                 and receipt.get("execution_complete") is True,
                 "fit_receipt", "fit is not complete")
        _require(receipt.get("native_execution") is True, "fit_receipt",
                 "the derivative fit did not execute natively")
        self._development_open = True


# ---------------------------------------------------------------------------
# Authorization and native admission gates


def verify_derivative_authorization(protocol: DerivativeProtocol, output: Path,
                                    authorization_path: str | Path | None,
                                    *, allow_existing_output: bool = False
                                    ) -> dict[str, Any]:
    """Refuse before any model import/load unless every binding is exact."""
    missing = _missing_bindings(protocol.config, protocol.report,
                                None if authorization_path is None
                                else Path(authorization_path), protocol.root)
    if missing:
        raise Refused("launch", "refused before model load: "
                      + ", ".join(dict.fromkeys(missing)))
    try:
        receipt = _validate_authorization(
            protocol.config, Path(authorization_path), output, protocol.root,
            protocol.report["implementation_sha256"], config_path=protocol.path,
            allow_existing_output=allow_existing_output)
    except PreflightError as error:
        raise Refused(error.check, error.detail) from error
    _require(receipt["execution_format"] == DERIVATIVE_EXECUTION_FORMAT,
             "authorization", "execution format is not the derivative identity")
    _require(receipt["source_identity"] == "dequantized_fp8_derivative",
             "authorization", "authorization does not bind the derivative source")
    return receipt


def _require_native_environment() -> None:
    if os.environ.get("REPLICA_CAPACITY_GUARD") != "1":
        raise Refused("capacity_guard",
                      "native execution requires scripts/run_capacity_preflight.py")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise Refused("cpu_fallback", "PYTORCH_ENABLE_MPS_FALLBACK=0 is required")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise Refused("offline", f"{name}=1 is required")


def _load_derivative_model(source_root: Path):
    """Import/load the native model; isolated so refusals can be proven cheap."""
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        str(source_root), local_files_only=True, trust_remote_code=True,
        torch_dtype="auto", attn_implementation="eager", use_safetensors=True)


def build_native_derivative_backend(protocol: DerivativeProtocol, *,
                                    authorization: Mapping[str, Any],
                                    device: str) -> Backend:
    _require(device == "mps", "native_device",
             "derivative BF16 execution requires --device mps")
    config = protocol.derivative_config
    precision = protocol.precision
    source = protocol.source
    source_root = Path(source["source_root"])
    _require(sha256_file(source_root / "config.json") == source["config_sha256"],
             "source", "derivative config bytes changed after preflight")
    admission = evaluate_derivative_p1_spine(
        precision, config, device=device, spine_sites=P1_SPINE_SITES,
        spine_slots=P1_SPINE_SLOTS, memory_size=P1_MEMORY_SIZE,
        heads=P1_PORT_HEADS)
    _require(admission["admitted"], "derivative_p1_spine",
             "; ".join(admission["reasons"]))
    _require(admission["precision_kind"] == DEQUANTIZED_FP8_DERIVATIVE,
             "precision_substitution_refused",
             "admission did not preserve the derivative identity")
    for filename, expected in (
        ("tokenizer.json", protocol.config["runtime"]["tokenizer"]["tokenizer_json_sha256"]),
        ("tokenizer_config.json", protocol.config["runtime"]["tokenizer"]["tokenizer_config_sha256"]),
        ("chat_template.jinja", protocol.config["runtime"]["tokenizer"]["chat_template_sha256"]),
    ):
        path = source_root / filename
        _require(path.is_file() and sha256_file(path) == expected, "tokenizer",
                 f"derivative tokenizer asset {filename} differs")
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise Refused("native_runtime", "transformers is unavailable") from error
    tokenizer = AutoTokenizer.from_pretrained(
        str(source_root), local_files_only=True, trust_remote_code=True)
    _require(getattr(tokenizer, "is_fast", False), "tokenizer",
             "fast tokenizer required")
    tokenizer_config = read_json(source_root / "tokenizer_config.json")
    template = (source_root / "chat_template.jinja").read_text(encoding="utf-8")
    tokenizer_json = (source_root / "tokenizer.json").read_text(encoding="utf-8")
    identity = load_tokenizer_identity(
        config, tokenizer_config, template, tokenizer_json=tokenizer_json)
    _require(identity.vocab_size == config.geometry.vocab_size, "tokenizer",
             "vocabulary differs")
    mode = resolve_thinking_mode(identity, "none")
    base = _load_derivative_model(source_root)
    _require(getattr(base.config, "model_type", None) == "spark2_5", "source",
             "loaded model is not spark2_5")
    adapter = Spark2_5Adapter(
        base, config, precision, device=device, require_official_geometry=True,
        projector_in_dim=int(config.geometry.hidden_size), zero_init_output=True,
        memory_size=P1_MEMORY_SIZE, heads=P1_PORT_HEADS,
        spine_sites=tuple(P1_SPINE_SITES), spine_slots=P1_SPINE_SLOTS)
    adapter.extension.spine.codec_version = TEXT_CODEC_VERSION
    adapter.extension.spine.representation_version = TEXT_REPRESENTATION_VERSION
    adapter.eval()
    adapter.seal_original_identity()
    stop_ids = tuple(sorted({int(getattr(tokenizer, "eos_token_id", 1)),
                             int(getattr(tokenizer, "pad_token_id", 2))}))
    return Backend(label=NATIVE_LABEL, adapter=adapter, tokenizer=tokenizer,
                   device=device, native_admissible=True, chat_mode=mode,
                   stop_token_ids=stop_ids, authorization=dict(authorization))


# ---------------------------------------------------------------------------
# Site justification


def review_site_request(sites: Sequence[int], *,
                        early_report: Mapping[str, Any] | None = None,
                        artifact_sha256: str | None = None,
                        compare_late: bool = False) -> dict[str, Any]:
    """Only early 3/7 runs here; 23/31 needs a completed matched early result."""
    requested = tuple(int(site) for site in sites)
    if requested == tuple(P1_SPINE_SITES):
        return {"sites": list(requested), "early_result_required": False,
                "late_comparison": "not_requested"}
    if requested == (23, 31):
        _require(compare_late, "late_sites",
                 "matched late 23/31 requires an explicit comparison request")
        _require(isinstance(early_report, Mapping), "late_sites",
                 "matched late 23/31 requires a completed early 3/7 report")
        _require(early_report.get("status") == "complete", "late_sites",
                 "matched late 23/31 requires a completed early 3/7 run")
        _require(artifact_sha256 is not None
                 and early_report.get("source", {}).get("artifact_sha256")
                 == artifact_sha256, "late_sites",
                 "late sites must use the same 36-layer derivative source")
        _require(early_report.get("sites") == list(P1_SPINE_SITES), "late_sites",
                 "the early report was not the 3/7 P1 result")
        return {"sites": list(requested), "early_result_required": True,
                "late_comparison": "matched_same_source_after_early_result"}
    raise Refused("spine_sites",
                  f"sites {requested} are neither early 3/7 nor matched late 23/31")


# ---------------------------------------------------------------------------
# Extension-only export/reload


def export_extension(adapter: Spark2_5Adapter, output: Path) -> dict[str, Any]:
    target = output / "extension_export"
    _require(not target.exists(), "extension_export",
             "extension export target must be fresh")
    manifest = adapter.save_extension(target)
    _require(manifest.get("contains_base_weights") is False
             and manifest.get("base_saved") is False, "extension_export",
             "the export copied base weights")
    return {
        "path": str(target), "manifest_sha256": _sha256(target / "manifest.json"),
        "weights_sha256": manifest["weights_sha256"],
        "contains_base_weights": False, "base_saved": False,
        "optimizer_saved": False,
        "fresh_process_reload_required": True,
    }


def load_extension_receipt(adapter: Spark2_5Adapter,
                           export_dir: str | Path) -> dict[str, Any]:
    export_dir = Path(export_dir)
    _require(export_dir.is_dir(), "extension_reload",
             f"extension export is missing at {export_dir}")
    manifest = read_json(export_dir / "manifest.json")
    receipt = adapter.load_extension(export_dir)
    return {
        "format": EXTENSION_RELOAD_FORMAT,
        "extension_only": receipt.get("extension_only") is True,
        "loaded_tensors": int(receipt["loaded_tensors"]),
        "loaded_parameter_elements": int(receipt["loaded_parameter_elements"]),
        "weights_sha256": receipt["weights_sha256"],
        "manifest_contains_base_weights":
            manifest.get("contains_base_weights") is True,
        "base_saved": manifest.get("base_saved") is True,
        "base_unchanged": adapter.base_unchanged() is True,
        "fresh_process_comparison_required":
            receipt.get("fresh_process_comparison_required") is True,
        "export_path": str(export_dir),
        "written_at_utc": _utc(),
    }


# ---------------------------------------------------------------------------
# Experiment-scope gates over the reused v2 fit/development artifacts


def _all_finite(values: Mapping[str, Any]) -> bool:
    if not values:
        return False
    for value in values.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        if not math.isfinite(float(value)):
            return False
    return True


def _reader_port_gradient_gate(port: Mapping[str, Any]) -> dict[str, bool]:
    """Intended reader ports are the memory-bound output projections.

    A P-only/full-context arm does not connect the spine/workspace port
    surfaces, so those tensors legitimately carry no gradient.  Every reported
    norm must still be finite and the intended memory reader ports must have
    received a finite, nonzero gradient.
    """
    norms = port.get("gradient_norms") or {}
    intended = {name: value for name, value in norms.items()
                if ".ports.memory." in name and name.endswith(".output_projection.weight")}
    finite_all = _all_finite(norms)
    intended_finite = _all_finite(intended)
    intended_nonzero = intended_finite and all(float(value) > 0.0
                                               for value in intended.values())
    unexpected_missing = [
        name for name in port.get("missing_gradient_names") or []
        if ".ports.memory." in name]
    return {
        "finite_all_port_norms": finite_all,
        "intended_reader_ports_present": bool(intended),
        "intended_reader_gradients_finite": intended_finite,
        "intended_reader_gradients_nonzero": intended_nonzero,
        "intended_reader_ports_without_gradient": not unexpected_missing,
    }


def _reader_port_contract_ok(gate: Mapping[str, bool]) -> bool:
    """The finite/present contract; a zero gradient is still a real gradient."""
    return all(value for key, value in gate.items()
               if key != "intended_reader_gradients_nonzero")


def verify_derivative_gates(protocol: DerivativeProtocol, output: Path,
                            fit_receipt: Mapping[str, Any],
                            *, backend: Backend | None = None) -> dict[str, Any]:
    rows = _read_jsonl(output / "fit/trajectories.jsonl", "fit_trajectories")
    _require(len(rows) == int(protocol.schedule["fit_events"]), "fit_gates",
             "fit trajectory count differs")
    finite_p = finite_ports = moments = clock = ownership = True
    p_changed = port_changed = True
    no_non_port_changes = True
    port_evidence: list[dict[str, bool]] = []
    for row in rows:
        learned = row["learned"]
        finite_p = finite_p and _all_finite(learned["gradients"]["norms"])
        ports = learned["port_optimizer"]
        gate = _reader_port_gradient_gate(ports)
        port_evidence.append(gate)
        finite_ports = finite_ports and _reader_port_contract_ok(gate)
        moments = moments and bool(learned["moments"]["changed_keys"])
        clock = clock and bool(learned["clock"]["eligible_events_advanced"]
                               and learned["clock"]["revision_advanced"])
        ownership = ownership and bool(
            learned["optimizer_ownership"]["single_p_owner"]
            and learned["optimizer_ownership"]["physical_optimizer_steps"] >= 1
            and ports["stepped"] and ports["physical_steps"] >= 1)
        p_changed = p_changed and bool(learned["actual_parameter_update"]) and bool(
            learned["changed_parameter_names"]) and set(
                learned["changed_parameter_names"]) <= {"gate", "up", "down"}
        port_changed = port_changed and bool(ports["changed_parameter_names"]) and all(
            name.startswith("extension.spine.ports.")
            for name in ports["changed_parameter_names"])
        no_non_port_changes = (no_non_port_changes
                               and not learned["non_port_spine_parameter_changes"])
    parity = fit_receipt["initial_null_parity"]["exact_null_parity"] is True
    base_frozen = True if backend is None else bool(backend.adapter.base_unchanged())
    gates = {
        "zero_init_parity": parity,
        "finite_gradients_into_P": finite_p,
        "finite_gradients_into_reader_ports": finite_ports,
        "selected_P_actually_changed": p_changed,
        "reader_ports_actually_changed": port_changed,
        "moments_recorded": moments,
        "learning_clock_and_revision_advanced": clock,
        "single_optimizer_ownership": ownership,
        "no_non_port_spine_parameter_changes": no_non_port_changes,
        "frozen_base_integrity": base_frozen,
        "selected_P_optimizer_steps": int(fit_receipt["selected_P_optimizer_steps"]),
        "response_port_optimizer_steps": int(
            fit_receipt["response_port_optimizer_steps"]),
        "fit_events": len(rows),
    }
    _require(all(bool(value) for key, value in gates.items()
                 if key not in {"selected_P_optimizer_steps",
                                "response_port_optimizer_steps", "fit_events"}),
             "experiment_scope_gates",
             "failed gates: " + ", ".join(
                 key for key, value in gates.items()
                 if isinstance(value, bool) and not value))
    return gates


def _full_context_first(output: Path) -> dict[str, Any]:
    rows = _read_jsonl(output / "development/trajectories.jsonl",
                       "development_trajectories")
    _require(bool(rows), "full_context_first", "no development rows were recorded")
    first = rows[0]
    _require(first.get("execution_complete") is True
             and first.get("arm_id") == "full_eligible_context", "full_context_first",
             "the untouched full-context arm was not completed first")
    return {
        "first_arm": first["arm_id"],
        "untouched_baseline": True,
        "completed_before_P_interpretation": True,
        "development_rows": len(rows),
        "development_cases_per_arm": 12,
        "arms": list(ARMS),
    }


# ---------------------------------------------------------------------------
# CPU synthetic derivative smoke


def _smoke_parent_and_derivative(seed: int, *, layers: int = 8):
    raw = contextual_config_raw(layers=layers, vocab_size=256,
                                max_position_embeddings=64)
    parent_raw = copy.deepcopy(raw)
    parent_raw["quantization_config"] = {
        "quant_method": "compressed-tensors", "format": "float-quantized",
        "version": "0.16.0", "kv_cache_scheme": None,
        "config_groups": {"group_0": {
            "format": "float-quantized",
            "targets": ["re:.*self_attn.q_k_v_proj$",
                        "re:.*self_attn.out_proj$",
                        "re:.*mlp.(gate_proj|up_proj|gate_up_proj|down_proj)$"],
            "weights": {"strategy": "channel", "num_bits": 8,
                        "dynamic": False, "symmetric": True, "type": "float"},
            "input_activations": {"strategy": "token", "num_bits": 8,
                                  "dynamic": True, "symmetric": True, "type": "float"},
        }},
        "ignore": [f"model.layers.{index}.self_attn.g_proj"
                   for index in range(layers)] + ["lm_head"],
    }
    parent_config = load_spark2_5_config(parent_raw, require_official_geometry=False)
    parent = official_native_fp8_precision(
        parent_config,
        artifact_sha256=publication.sha256_bytes(b"replica-synthetic-fp8-parent-v1"))
    dequantization_sha256 = sha256_file(ROOT / "scripts" / "dequantize_spark_fp8.py")
    artifact_sha256 = publication.sha256_bytes(
        b"replica-synthetic-dequantized-derivative-artifact-v1")
    raw["replica_precision"] = {
        "kind": DEQUANTIZED_FP8_DERIVATIVE,
        "parent_kind": OFFICIAL_NATIVE_FP8,
        "parent_artifact_sha256": parent.artifact_sha256,
        "dequantization_sha256": dequantization_sha256,
        "activation_semantics": DERIVATIVE_ACTIVATION_SEMANTICS,
        "equivalent_to_bf16_release": False,
    }
    config = load_spark2_5_config(raw, require_official_geometry=False)
    precision = dequantized_fp8_derivative(
        config, parent=parent, artifact_sha256=artifact_sha256,
        dequantization_sha256=dequantization_sha256,
        activation_semantics=DERIVATIVE_ACTIVATION_SEMANTICS, storage="bf16")
    return raw, config, precision, parent


def _smoke_adapter(config, precision, seed: int) -> Spark2_5Adapter:
    torch.manual_seed(seed)
    base = ContextualSparkCausalLM(config)
    adapter = Spark2_5Adapter(
        base, config, precision, device="cpu", require_official_geometry=False,
        projector_layer=3, memory_size=8, heads=2, spine_sites=(3, 7),
        spine_slots=4)
    adapter.seal_original_identity()
    return adapter


def run_cpu_smoke(config_path: Path, output: Path, *,
                  full_hash: bool = False) -> dict[str, Any]:
    _require(not output.exists(), "output", "smoke output must be fresh")
    protocol = DerivativeProtocol.load(config_path, full_hash=full_hash)
    seed = int(protocol.config["seed"])
    raw, config, precision, parent = _smoke_parent_and_derivative(seed)
    adapter = _smoke_adapter(config, precision, seed)
    ids, mask = ids_and_mask()
    with torch.no_grad():
        base_logits = adapter.base(input_ids=ids, attention_mask=mask,
                                   use_cache=False).logits
        adapter_logits = adapter.forward_text(ids, mask)
    parity = torch.equal(adapter_logits, base_logits)
    _require(parity, "zero_init_parity",
             "the derivative fixture adapter did not reproduce base logits")

    experiment = SpineMemoryExperiment(
        spec=adapter.spine_spec, core=adapter.extension.spine,
        bank_experts=1, bank_hidden_width=8, seed=seed, read_records=3,
        scope="derivative-cpu-smoke")
    experiment.bank.lr = 1e-2
    path = SparkTextMemoryPath(adapter, experiment, response_lr=1e-2)
    open_memory_ports(path)
    populate_contextual_evidence(path, SMOKE_RECORDS)
    labels = labels_from_ids(ids)
    query = prepare_arm_query(
        path, arm="p_only", question="which token follows the prefix",
        input_ids=ids, attention_mask=torch.ones_like(ids), expert_id=0)
    port_optimizer = build_port_optimizer(path, lr=1e-2)
    step = direct_response_ce_step(
        path, query, labels, event_id="derivative-cpu-smoke-0",
        port_optimizer=port_optimizer)

    finite_p = _all_finite(step["gradients"]["norms"])
    port_evidence = _reader_port_gradient_gate(step["port_optimizer"])
    finite_ports = _reader_port_contract_ok(port_evidence)
    _require(finite_p and finite_ports, "gradients",
             "smoke gradients into P or intended reader ports were not finite")
    _require(step["actual_parameter_update"] is True
             and set(step["changed_parameter_names"]) <= {"gate", "up", "down"},
             "selected_P_update", "the smoke did not actually update selected P")
    _require(bool(step["moments"]["changed_keys"]), "moments",
             "the smoke update left NativeFusedAdamW moments unchanged")
    _require(bool(step["clock"]["eligible_events_advanced"]
                  and step["clock"]["revision_advanced"]), "clock",
             "the smoke update did not advance the learning clock")
    _require(step["optimizer_ownership"]["single_p_owner"] is True
             and step["port_optimizer"]["stepped"] is True
             and step["port_optimizer"]["physical_steps"] == 1, "optimizer",
             "the smoke did not perform one bounded dual-owner update")
    _require(not step["non_port_spine_parameter_changes"], "scope",
             "the smoke changed a non-port spine parameter")
    _require(adapter.base_unchanged() is True, "frozen_base",
             "the smoke changed the frozen original backbone")

    output.mkdir(parents=True, exist_ok=False)
    export = export_extension(adapter, output)
    fresh = _smoke_adapter(config, precision, seed)
    reload_receipt = load_extension_receipt(fresh, export["path"])
    _require(reload_receipt["extension_only"] is True
             and reload_receipt["loaded_tensors"] == len(
                 list(fresh.new_named_parameters()))
             and reload_receipt["base_unchanged"] is True, "extension_reload",
             "extension-only reload did not restore every extension tensor")
    with torch.no_grad():
        trained_logits = adapter.forward_text(ids, mask)
        reloaded_logits = fresh.forward_text(ids, mask)
    reload_exact = torch.equal(trained_logits, reloaded_logits)
    _require(reload_exact, "extension_reload",
             "reloaded extension logits differ from the trained extension")

    substitution_refusals: dict[str, str] = {}
    for name, call in (
        ("official_bf16_label_on_derivative_config",
         lambda: official_bf16_precision(config)),
        ("official_fp8_label_on_derivative_config",
         lambda: official_native_fp8_precision(config)),
        ("classifier_without_derivative_manifest",
         lambda: classify_precision(config)),
    ):
        try:
            call()
        except Spark2_5ContractError:
            substitution_refusals[name] = "refused"
        else:
            raise Refused("precision_substitution_refused",
                          f"{name} was not refused")
    official_raw = copy.deepcopy(raw)
    official_raw.pop("replica_precision")
    official_config = load_spark2_5_config(
        official_raw, require_official_geometry=False)
    official_identity = official_bf16_precision(
        official_config, artifact_sha256="a" * 64)
    try:
        Spark2_5Adapter(ContextualSparkCausalLM(official_config), config,
                        official_identity, device="cpu",
                        require_official_geometry=False)
    except SparkAdapterError:
        substitution_refusals["official_identity_on_derivative_config"] = "refused"
    else:
        raise Refused("precision_substitution_refused",
                      "an official identity loaded the derivative config")
    stale_refusals: dict[str, str] = {}
    try:
        dequantized_fp8_derivative(
            config, parent=parent, artifact_sha256=parent.artifact_sha256,
            dequantization_sha256=precision.dequantization_sha256,
            activation_semantics=DERIVATIVE_ACTIVATION_SEMANTICS, storage="bf16")
    except Spark2_5ContractError:
        stale_refusals["parent_artifact_reuse"] = "refused"
    else:
        raise Refused("stale_hashes", "the derivative reused the parent artifact")
    stale_precision = replace(precision, dequantization_sha256="0" * 64)
    try:
        Spark2_5Adapter(ContextualSparkCausalLM(config), config,
                        stale_precision, device="cpu",
                        require_official_geometry=False)
    except (SparkAdapterError, Spark2_5ContractError):
        stale_refusals["stale_dequantization_hash"] = "refused"
    else:
        raise Refused("stale_hashes",
                      "a stale dequantization hash was admitted")

    report = {
        "format": SMOKE_FORMAT,
        "status": "smoke_passed",
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.sha256,
        "backend": SMOKE_LABEL,
        "engineering_only": True,
        "native_execution": False,
        "native_model_loaded": False,
        "fixture_model_constructed": True,
        "downloads": 0,
        "precision_kind": precision.kind,
        "execution_format": DERIVATIVE_EXECUTION_FORMAT,
        "precision_digest": precision.digest,
        "synthetic_parent_identity": parent.kind,
        "seed": seed,
        "checks": {
            "derivative_config_and_manifest_preflight": True,
            "zero_init_parity_exact_logits": bool(parity),
            "finite_gradients_into_P": bool(finite_p),
            "finite_gradients_into_reader_ports": bool(finite_ports),
            "intended_reader_port_evidence": port_evidence,
            "selected_P_changed": True,
            "reader_ports_changed": True,
            "native_fused_adamw_updates": int(
                step["optimizer_ownership"]["physical_optimizer_steps"]
                + step["port_optimizer"]["physical_steps"]),
            "moments_changed_keys": len(step["moments"]["changed_keys"]),
            "learning_clock_advanced": True,
            "frozen_base_integrity": bool(adapter.base_unchanged()),
            "extension_only_reload": bool(reload_receipt["extension_only"]),
            "extension_reload_logits_exact": bool(reload_exact),
            "same_process_new_adapter": True,
            "official_fp8_or_bf16_substitution_refused":
                len(substitution_refusals) == 4,
            "stale_hashes_refused": len(stale_refusals) == 2,
        },
        "substitution_refusals": substitution_refusals,
        "stale_hash_refusals": stale_refusals,
        "extension_reload": reload_receipt,
        "optimizer": {
            "owner": "expert_bank.optimizer",
            "port_owner": "extension.spine.ports.*.output_projection",
            "physical_P_steps": int(
                step["optimizer_ownership"]["physical_optimizer_steps"]),
            "physical_port_steps": int(step["port_optimizer"]["physical_steps"]),
            "changed_parameters": list(step["changed_parameter_names"]),
            "changed_port_parameters": list(
                step["port_optimizer"]["changed_parameter_names"]),
        },
        "site_review": review_site_request(P1_SPINE_SITES),
        "late_sites": "deferred_until_completed_matched_early_3_7_result",
        "non_claims": [
            "this is a synthetic CPU engineering fixture, not Spark native evidence",
            "no useful learning, accuracy, speed or RSI claim follows from the smoke",
            "the smoke does not open sealed-final data or execute official FP8/BF16",
        ],
        "written_at_utc": _utc(),
    }
    _write_json(output, "smoke_report.json", report)
    return report


# ---------------------------------------------------------------------------
# Native isolated-process replay


def _restart_command(protocol: DerivativeProtocol, output: Path, prepared: Path,
                     authorization: Path, export_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(protocol.root / "scripts/run_spark_persistent_learner_derivative_bf16.py"),
        "--config", str(protocol.path), "--prepared", str(prepared),
        "--output", str(output), "--backend", NATIVE_LABEL,
        "--device", "mps", "--stage", "restart",
        "--authorization", str(authorization),
        "--extension-export", str(export_dir),
    ]


def _run_isolated_restart(protocol: DerivativeProtocol, output: Path,
                          prepared: Path, authorization: Path,
                          export_dir: Path, *, timeout: int) -> dict[str, Any]:
    restart_dir = output / "restart"
    restart_dir.mkdir(parents=True, exist_ok=True)
    command = _restart_command(protocol, output, prepared, authorization, export_dir)
    _write_json(output, "restart/child_command.json",
                {"command": command, "written_at_utc": _utc()})
    child = subprocess.run(
        command, cwd=protocol.root, text=True, capture_output=True, timeout=timeout,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
             "PYTORCH_ENABLE_MPS_FALLBACK": "0", "HF_HUB_OFFLINE": "1",
             "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
             "REPLICA_NATIVE_SUPERVISED_CHILD": "1"})
    (restart_dir / "child.stdout").write_text(child.stdout, encoding="utf-8")
    (restart_dir / "child.stderr").write_text(child.stderr, encoding="utf-8")
    _require(child.returncode == 0, "restart",
             "fresh-process derivative restart failed: "
             + child.stderr[-2000:])
    report = _read_json(restart_dir / "restart_report.json", "restart")
    reload_receipt = _read_json(restart_dir / "extension_reload.json",
                                "extension_reload")
    return {"report": report, "extension_reload": reload_receipt}


# ---------------------------------------------------------------------------
# Native protocol execution


def _native_impl(protocol: DerivativeProtocol, dataset: DerivativeDataset,
                 output: Path, *, stage: str, authorization_path: Path,
                 extension_export: Path) -> dict[str, Any]:
    started = time.monotonic()
    fresh = stage in {"all", "fit"}
    _require(not output.exists() if fresh else output.is_dir(), "output",
             "fit output must be fresh" if fresh else "resume output is missing")
    receipt = verify_derivative_authorization(
        protocol, output, authorization_path,
        allow_existing_output=not fresh)
    backend = build_native_derivative_backend(
        protocol, authorization=receipt, device="mps")
    context = int(protocol.config["runtime"]["context_tokens_max"])
    new_tokens = int(protocol.config["runtime"]["generation"]["max_new_tokens"])
    ledger = WorkLedger(protocol, context_tokens=context, max_new_tokens=new_tokens)
    meter = publication.LiveByteMeter(
        output, label="spark_persistent_learner_derivative_bf16",
        cap_bytes=int(protocol.schedule["artifact_bytes_max"]))
    ledger.charge("model_loads")
    ledger.charge("tokenizer_loads")
    try:
        if stage == "fit":
            fit = run_fit(protocol, dataset, backend, output, ledger, meter)
            ledger.set_measured("artifact_bytes", meter.peak_bytes)
            report = {
                "format": RUN_FORMAT, "stage": "fit", "status": "fit_executed",
                "protocol_id": protocol.protocol_id,
                "protocol_sha256": protocol.sha256, "fit": fit.receipt,
                "work": ledger.report(),
            }
            _write_json(output, "run_report.json", report, meter=meter)
            return report
        if stage == "development":
            fit = load_fit_state(protocol, dataset, backend, output)
            development = run_development(
                protocol, dataset, backend, output, ledger, meter, fit)
            report = {
                "format": RUN_FORMAT, "stage": "development",
                "status": "development_executed",
                "protocol_id": protocol.protocol_id,
                "protocol_sha256": protocol.sha256, "development": development,
                "full_context_first": _full_context_first(output),
                "work": ledger.report(),
            }
            _write_json(output, "run_report.json", report, meter=meter)
            return report
        if stage == "restart":
            reload_receipt = load_extension_receipt(
                backend.adapter, extension_export)
            restart = run_restart(protocol, dataset, backend, output, ledger, meter)
            _write_json(output, "restart/extension_reload.json",
                        reload_receipt, meter=meter)
            ledger.set_measured("artifact_bytes", meter.peak_bytes)
            report = {
                "format": RUN_FORMAT, "stage": "restart",
                "status": "restart_executed",
                "protocol_id": protocol.protocol_id,
                "protocol_sha256": protocol.sha256,
                "restart": restart, "extension_reload": reload_receipt,
                "base_unchanged": backend.adapter.base_unchanged() is True,
                "work": ledger.report(),
            }
            _write_json(output, "run_report.json", report, meter=meter)
            return report

        fit = run_fit(protocol, dataset, backend, output, ledger, meter)
        gates = verify_derivative_gates(
            protocol, output, fit.receipt, backend=backend)
        development = run_development(
            protocol, dataset, backend, output, ledger, meter, fit)
        full_context = _full_context_first(output)
        export = export_extension(backend.adapter, output)
        base_frozen = backend.adapter.base_unchanged() is True
        _require(base_frozen, "frozen_base_integrity",
                 "the original backbone changed before the restart boundary")
        fit_receipt = fit.receipt
        fit = None
        backend.close()
        backend = None
        gc.collect()
        isolated = _run_isolated_restart(
            protocol, output, dataset.root, authorization_path,
            Path(export["path"]),
            timeout=int(protocol.schedule["wall_seconds_max"]))
        _merge_restart_report(ledger, isolated["report"])
        ledger.finish()
        reload_receipt = isolated["extension_reload"]
        _require(reload_receipt["extension_only"] is True
                 and reload_receipt["base_unchanged"] is True, "extension_reload",
                 "the fresh-process extension reload receipt is invalid")
        evidence = {
            **gates,
            "full_context_first": full_context["first_arm"],
            "fresh_process_extension_reload":
                reload_receipt["extension_only"] is True,
            "fresh_process_restart_parity":
                isolated["report"]["parity"]["exact_token_ids"] is True,
            "restart_P_state_preserved":
                isolated["report"].get("P_state_preserved") is True,
            "base_frozen_through_restart":
                reload_receipt["base_unchanged"] is True,
            "extension_export": export,
            "extension_reload": reload_receipt,
        }
        _require(all(bool(value) for key, value in evidence.items()
                     if isinstance(value, bool)), "experiment_scope_gates",
                 "failed native gates: " + ", ".join(
                     key for key, value in evidence.items()
                     if isinstance(value, bool) and not value))
        report = {
            "format": RUN_FORMAT,
            "protocol_id": protocol.protocol_id,
            "protocol_sha256": protocol.sha256,
            "status": "complete",
            "stage": "all",
            "backend": NATIVE_LABEL,
            "native_execution": True,
            "model_loaded": True,
            "execution_format": DERIVATIVE_EXECUTION_FORMAT,
            "precision_kind": protocol.precision.kind,
            "precision_digest": protocol.precision.digest,
            "artifact_sha256": protocol.source["artifact_sha256"],
            "source": {
                "source_identity": "dequantized_fp8_derivative",
                "artifact_sha256": protocol.source["artifact_sha256"],
                "config_sha256": protocol.source["config_sha256"],
                "manifest_sha256": protocol.source["manifest_sha256"],
                "precision_digest": protocol.source["precision_digest"],
                "model_identity": protocol.source["precision_digest"],
            },
            "authorization_id": receipt["authorization_id"],
            "attempt_id": receipt["attempt_id"],
            "protocol_commit": protocol.config["launch"]["protocol_commit"],
            "implementation_sha256": dict(
                protocol.report["implementation_sha256"]),
            "sites": list(P1_SPINE_SITES),
            "late_sites": "deferred_until_completed_matched_early_3_7_result",
            "fit": fit_receipt,
            "development": development,
            "restart": isolated["report"],
            "evidence": evidence,
            "learning_claim": "none_engineering_only",
            "sealed_final_opened": False,
            "official_fp8_unchanged": True,
            "official_bf16_unchanged": True,
            "work": ledger.report(),
            "artifact": meter.report(),
            "wall_seconds": time.monotonic() - started,
            "non_claims": [
                "no useful learning or native qualification is claimed by execution",
                "the derivative does not restore quantization loss or equal official BF16",
                "official FP8 and official BF16 remain separate unsubstituted identities",
            ],
            "written_at_utc": _utc(),
        }
        report["storage_artifacts"] = _write_storage_receipts(output, meter)
        for _ in range(3):
            ledger.set_measured("artifact_bytes", meter.peak_bytes)
            report["work"] = ledger.report()
            report["artifact"] = meter.report()
            _write_json(output, "run_report.json", report, meter=meter)
            meter.observe()
        return report
    except Exception as error:
        output.mkdir(parents=True, exist_ok=True)
        failure = {
            "format": "replica.spark_persistent_learner_derivative.failure.v1",
            "stage": stage, "backend": NATIVE_LABEL, "error": str(error),
            "error_type": type(error).__name__, "traceback": traceback.format_exc(),
            "work": ledger.report(), "native_execution": False,
            "preserved_partial_artifacts": True,
            "official_fp8_substitution": False,
            "official_bf16_substitution": False,
            "written_at_utc": _utc(),
        }
        try:
            _write_json(output, "failure.json", failure)
        except Exception:
            pass
        raise
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass


def run_derivative_protocol(config_path: str | Path = CONFIG_PATH,
                            prepared_root: str | Path = "data/spark_persistent_learner_v2",
                            output_root: str | Path = "",
                            *, backend_kind: str = NATIVE_LABEL,
                            device: str = "mps", stage: str = "all",
                            authorization_path: str | Path | None = None,
                            extension_export: str | Path | None = None,
                            full_hash: bool = True,
                            allow_existing_output: bool = False) -> dict[str, Any]:
    if backend_kind == SMOKE_LABEL:
        _require(stage == "smoke", "stage", "the CPU smoke supports only --stage smoke")
        _require(device == "cpu", "device", "the CPU smoke requires --device cpu")
        return run_cpu_smoke(Path(config_path).resolve(),
                             Path(output_root).resolve(), full_hash=False)
    _require(backend_kind == NATIVE_LABEL, "backend",
             f"unknown derivative backend {backend_kind}")
    _require(device == "mps", "native_device",
             "the derivative native path requires --device mps")
    _require(stage in NATIVE_STAGES, "stage", f"unknown stage {stage}")
    # The cheap binding pass runs first so a missing/incorrect authorization is
    # refused before any multi-gigabyte shard hashing or model import.
    protocol = DerivativeProtocol.load(config_path, full_hash=False)
    output = Path(output_root).resolve()
    path = None if authorization_path is None else Path(authorization_path)
    verify_derivative_authorization(
        protocol, output, path,
        allow_existing_output=allow_existing_output or stage not in {"all", "fit"})
    _require(full_hash, "source",
             "native execution requires full derivative shard hashing; "
             "--skip-full-hash is smoke/preflight only")
    try:
        full_report = validate_config(protocol.config, protocol.root, full_hash=True)
    except PreflightError as error:
        raise Refused(error.check, error.detail) from error
    _require(full_report["source"]["artifact_sha256"]
             == protocol.source["artifact_sha256"], "source",
             "full-hash source artifact differs from the review binding")
    _require(full_report["source"]["precision_digest"]
             == protocol.source["precision_digest"], "source",
             "full-hash precision digest differs from the review binding")
    _require(full_report["implementation_sha256"]
             == protocol.report["implementation_sha256"], "implementation",
             "implementation bytes changed during authorization")
    protocol = replace(protocol, report=full_report)
    _require_native_environment()
    dataset = DerivativeDataset.load(
        prepared_root,
        expected_sha256=protocol.config["dataset"].get("manifest_sha256"),
        protocol_sha256=protocol.sha256)
    export = (Path(extension_export).resolve() if extension_export is not None
              else output / "extension_export")
    supervised = os.environ.get("REPLICA_NATIVE_SUPERVISED_CHILD") == "1"
    if supervised:
        return _native_impl(
            protocol, dataset, output, stage=stage,
            authorization_path=Path(path), extension_export=export)
    with exclusive_mps_worker(protocol.root):
        return _native_impl(
            protocol, dataset, output, stage=stage,
            authorization_path=Path(path), extension_export=export)


# ---------------------------------------------------------------------------
# CLI


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--prepared", type=Path,
                        default=Path("data/spark_persistent_learner_v2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=[NATIVE_LABEL, SMOKE_LABEL],
                        default=NATIVE_LABEL)
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--stage", choices=list(STAGES), default="all")
    parser.add_argument("--authorization", type=Path, default=None)
    parser.add_argument("--extension-export", type=Path, default=None)
    parser.add_argument("--skip-full-hash", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run_derivative_protocol(
            args.config, args.prepared, args.output, backend_kind=args.backend,
            device=args.device, stage=args.stage,
            authorization_path=args.authorization,
            extension_export=args.extension_export,
            full_hash=not args.skip_full_hash)
        print(json.dumps(_json_safe(report), indent=2, sort_keys=True,
                         allow_nan=False))
        return 0
    except Exception as error:
        detail = (error.as_dict() if isinstance(error, Refused)
                  else {"check": "runner", "detail": str(error),
                        "traceback": traceback.format_exc()})
        print(json.dumps({"status": "refused", **detail}, indent=2,
                         sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DerivativeProtocol", "DerivativeDataset", "NATIVE_LABEL", "SMOKE_LABEL",
    "RUN_FORMAT", "SMOKE_FORMAT", "build_native_derivative_backend",
    "export_extension", "load_extension_receipt", "main",
    "review_site_request", "run_cpu_smoke", "run_derivative_protocol",
    "verify_derivative_authorization", "verify_derivative_gates",
]

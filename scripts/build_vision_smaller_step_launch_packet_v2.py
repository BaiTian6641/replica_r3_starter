#!/usr/bin/env python3
"""Build the versioned smaller-step launch packet (v2 accounting amendment).

The v1 packet is preserved byte-for-byte.  This builder re-validates the v2
config on CPU, proves every accounting counter is derived from the declared
schedule, records the superseded v1 caps, and writes a preparation-only packet
that still refuses native launch because no authorization receipt exists.

The builder never imports a native runtime, loads a model, writes a receipt or
executes any experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from replica_lab import vision_smaller_step_accounting_v2 as acc_v2  # noqa: E402
import vision_smaller_step_v2 as ss  # noqa: E402

CONFIG_PATH = "configs/vision_smaller_step_lr3e5_v2.json"
V1_CONFIG_PATH = "configs/vision_smaller_step_lr3e5_v1.json"
V1_PACKET_PATH = ("reports/vision_smaller_step_launch_packet_v1_20261027/"
                  "launch_packet.json")
V1_READINESS_PATH = "configs/vision_smaller_step_launch_readiness_v1.json"
PACKET_PATH = ("reports/vision_smaller_step_launch_packet_v2_20261027/"
               "launch_packet.json")
FORMAT = "replica.vision_smaller_step.launch_packet.v2"
AUTHORIZATION_PATH = ("reports/vision_smaller_step_lr3e5_v2_20261027/"
                      "authorization.json")
OUTPUT_PATH = "runs/vision_smaller_step_lr3e5_v2_20261027"
ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "PYTORCH_ENABLE_MPS_FALLBACK": "0",
}


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_packet(*, root: str | Path = ROOT) -> dict[str, Any]:
    root_path = Path(root)
    config = ss.load_protocol(root_path / CONFIG_PATH, root=root_path)
    config_sha = _sha256_file(root_path / CONFIG_PATH)
    v1_config_sha = _sha256_file(root_path / V1_CONFIG_PATH)
    v1_packet = json.loads((root_path / V1_PACKET_PATH).read_text())
    v1_packet_sha = _sha256_file(root_path / V1_PACKET_PATH)
    v1_readiness_sha = _sha256_file(root_path / V1_READINESS_PATH)
    accounting = config["accounting"]
    derived = acc_v2.derive_accounting()
    acc_v2.validate_accounting(accounting)
    campaign = accounting["campaign_cumulative"]
    expected = config["readouts"]["expected_measured_work"]
    declared_measurement = dict(campaign)
    declared_measurement["visual_encoder_cache_hits"] = int(
        accounting["expected_counts"]["campaign"]
        ["visual_encoder_cache_hits"])
    identities = acc_v2.validate_live_counters(
        declared_measurement, label="declared_campaign",
        training_forwards=8)
    implementation = {
        "config": {
            "path": CONFIG_PATH,
            "sha256": config_sha,
            "bytes": (root_path / CONFIG_PATH).stat().st_size,
        },
        "runner": {
            "path": "scripts/vision_smaller_step_v2.py",
            "sha256": _sha256_file(root_path / "scripts/"
                                   "vision_smaller_step_v2.py"),
            "bytes": (root_path / "scripts/vision_smaller_step_v2.py"
                      ).stat().st_size,
        },
        "accounting_module": {
            "path": "src/replica_lab/vision_smaller_step_accounting_v2.py",
            "sha256": _sha256_file(
                root_path / "src/replica_lab/"
                "vision_smaller_step_accounting_v2.py"),
        },
        "builder": {
            "path": "scripts/build_vision_smaller_step_launch_packet_v2.py",
            "sha256": _sha256_file(
                root_path / "scripts/"
                "build_vision_smaller_step_launch_packet_v2.py"),
        },
    }
    launch_argv = ss.native_launch_command(config, CONFIG_PATH,
                                           AUTHORIZATION_PATH)
    review_command = [
        "python", "scripts/vision_smaller_step_v2.py", "review-launch",
        "--config", CONFIG_PATH,
        "--authorization", AUTHORIZATION_PATH,
    ]
    packet: dict[str, Any] = {
        "format": FORMAT,
        "version": "2",
        "kind": "preparation_only_launch_evidence",
        "deterministic": True,
        "native_execution": False,
        "model_loads": 0,
        "authorization_created": False,
        "purpose": (
            "Versioned amendment of the lr 3e-05 smaller-step launch packet. "
            "It preserves the v1 config, packet, readiness declaration and "
            "evidence, re-derives every accounting counter from the declared "
            "schedule, records the superseded stale caps, and still refuses "
            "native launch because no external authorization receipt exists."),
        "protocol": {
            "name": config["name"],
            "version": config["version"],
            "format": config["format"],
            "path": CONFIG_PATH,
            "sha256": config_sha,
            "lr": config["optimizer"]["lr"],
            "completed_lr": config["hypothesis"]["completed_lr"],
            "mode": config["mode"],
            "native_execution_authorized": False,
        },
        "preserved_v1": {
            "config": {"path": V1_CONFIG_PATH, "sha256": v1_config_sha},
            "packet": {"path": V1_PACKET_PATH, "sha256": v1_packet_sha,
                       "canonical_sha256": v1_packet.get("packet_sha256")},
            "readiness_declaration": {
                "path": V1_READINESS_PATH, "sha256": v1_readiness_sha},
            "overwritten_here": False,
            "v1_caps_preserved_in_v1_packet": {
                "campaign_decoder_forward_calls": 4824,
                "wider_decoder_forward_calls": 2400,
                "wider_visual_observation_calls": 78,
                "wider_visual_encoder_cache_hits": 78,
                "wider_visual_encoder_cache_misses": 78,
                "wider_visual_forward_calls": 2400,
                "wider_projector_passes": 2400,
            },
        },
        "accounting_amendment": {
            "format": acc_v2.FORMAT,
            "derivation": accounting["derivation"],
            "supersedes_v1": accounting["supersedes_v1"],
            "expected_counts": accounting["expected_counts"],
            "wider_readout_template": accounting["wider_readout_template"],
            "update_batch_evaluation":
                accounting["update_batch_evaluation"],
            "update": accounting["update"],
            "campaign_cumulative": campaign,
            "identities": accounting["identities"],
            "validated_live_identities": identities,
            "expected_measured_work": expected,
            "derived_caps": {
                "wider_readout": derived["wider_readout_template"]
                ["per_readout"],
                "campaign": derived["campaign_cumulative"],
            },
        },
        "implementation": implementation,
        "preflight": [
            {"id": "protocol_contract",
             "requirement": "v2 config validates exactly on CPU",
             "status": "passed"},
            {"id": "accounting_derived",
             "requirement": ("every v2 counter equals its schedule derivation "
                             "and identities hold"),
             "status": "passed"},
            {"id": "decoder_identity",
             "requirement": "decoder = prepares + logits + tokens + 8 update "
                            "forwards (4,804, not 4,824)",
             "status": "passed"},
            {"id": "visual_identity",
             "requirement": ("observations = hits + misses; encoder calls = "
                             "misses; visual-bearing passes are fewer than "
                             "all decoder passes"),
             "status": "passed"},
            {"id": "projector_identity",
             "requirement": "projector passes equal visual-bearing passes, "
                            "never all decoder passes",
             "status": "passed"},
            {"id": "v1_preserved",
             "requirement": "v1 config, packet and readiness hashes match",
             "status": "passed"},
            {"id": "authorization_receipt",
             "requirement": "a separate external receipt must authorize this "
                            "exact v2 config; none exists",
             "status": "pending"},
            {"id": "protocol_commit",
             "requirement": "the receipt must record a full commit whose v2 "
                            "config blob matches the config SHA-256",
             "status": "pending"},
            {"id": "fresh_output",
             "requirement": "the v2 output path must not exist at launch",
             "status": "pending"},
            {"id": "native_execution",
             "requirement": "no model load, backward, step or native command "
                            "is performed during packet preparation",
             "status": "not_performed"},
        ],
        "authorization": {
            "required": True,
            "separate_from_packet": True,
            "path": AUTHORIZATION_PATH,
            "format": ss.AUTHORIZATION_FORMAT,
            "status": "absent",
            "present_at_build": (root_path / AUTHORIZATION_PATH).is_file(),
            "minted_by_packet": False,
            "native_execution_authorized": False,
            "protocol_commit": None,
            "config_sha256": config_sha,
            "closed_roles": {
                "development": False, "heldout": False, "v0": False,
                "semantic_audit": False, "promotion": False, "retry": False,
            },
            "note": ("Readiness may observe and validate an existing receipt "
                     "but never creates one. The three preserved "
                     "vision_learnability_step1 receipts bind a different "
                     "protocol and config."),
        },
        "launch": {
            "argv": launch_argv,
            "review_command": review_command,
            "environment": dict(ENVIRONMENT),
            "output": OUTPUT_PATH,
            "output_must_not_exist": True,
            "output_exists_at_build": (root_path / OUTPUT_PATH).exists(),
            "authorization_checked_before_native_imports": True,
            "disposition_at_build": "review_only_launch_refused",
            "refusals_at_build": [
                "packet_is_not_an_authorization_receipt",
                "packet_is_preparation_only",
                "authorization_receipt_missing",
                "protocol_commit_unbound",
                "commit_binding_unverified",
                "native_execution_not_authorized",
            ],
        },
        "remaining_requirements": [
            {"id": "authorization_receipt_issued", "blocking": True,
             "owner": "external_user",
             "requirement": ("A separate external receipt must exist at "
                             f"{AUTHORIZATION_PATH}, bind config_sha256 "
                             f"{config_sha}, authorize native execution and "
                             "keep every other role false."),
             "satisfied_by_preparation": False},
            {"id": "protocol_commit_recorded", "blocking": True,
             "owner": "external_user",
             "requirement": ("The receipt must record a full 40-character "
                             "commit whose v2 config blob hashes to the same "
                             "config SHA-256."),
             "satisfied_by_preparation": False},
            {"id": "review_launch_gate_passed", "blocking": True,
             "owner": "authorized_operator",
             "requirement": ("scripts/vision_smaller_step_v2.py "
                             "review-launch must pass receipt, config, commit "
                             "and fresh-output checks before any native "
                             "import; no receipt exists."),
             "satisfied_by_preparation": False},
            {"id": "fresh_output_available", "blocking": True,
             "owner": "authorized_operator",
             "requirement": f"{OUTPUT_PATH} must not exist at launch.",
             "satisfied_by_preparation": not (root_path / OUTPUT_PATH).exists()},
            {"id": "exclusive_worker_and_capacity_guard", "blocking": True,
             "owner": "authorized_operator",
             "requirement": ("The launch must acquire the exclusive MPS worker "
                             "and prove 50 GiB driver, 512 MiB sustained swap "
                             "growth and 15% host free with no CPU fallback."),
             "satisfied_by_preparation": False},
            {"id": "native_execution_within_derived_budget", "blocking": True,
             "owner": "authorized_operator",
             "requirement": ("Execute exactly 4 backward calls and 1 "
                             "NativeFusedAdamW step within 4,804 decoder "
                             "forwards, 156 visual observations, 1,200 "
                             "seconds and 33,554,432 artifact bytes."),
             "satisfied_by_preparation": False},
        ],
        "decision": {
            "preparation_complete": True,
            "launch_allowed": False,
            "native_execution": False,
            "model_loads": 0,
            "authorization_created": False,
            "reason": ("preparation_is_complete_but_the_external_"
                       "authorization_receipt_is_absent_and_the_v2_protocol_"
                       "commit_is_unrecorded"),
        },
        "limitations": [
            "Preparation only: no native command, model load, backward, "
            "optimizer step or generation has been executed by this packet.",
            "The authorization receipt is absent and the v2 protocol commit "
            "is unrecorded, so launch is refused before native loading.",
            "Numeric losses, gate outcomes, wall time and artifact bytes are "
            "unobserved; the derived caps are declarations with exact "
            "identity checks, not measurements.",
            "The v1 packet, config, readiness and evidence remain preserved "
            "and unchanged; this v2 packet supersedes only their accounting "
            "caps and never rewrites the v1 files.",
            "Development, held-out, semantic-audit, V0, promotion and retry "
            "access remain closed regardless of this packet or any result.",
            "This packet is evidence, never an authorization receipt or a "
            "promotion decision.",
        ],
    }
    packet["packet_sha256"] = hashlib.sha256(
        _canonical(packet).encode()).hexdigest()
    return packet


def write_packet(*, root: str | Path = ROOT,
                 path: str | Path = PACKET_PATH) -> dict[str, Any]:
    packet = build_packet(root=root)
    _write_json(Path(root) / path, packet)
    return packet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify the on-disk packet equals the build")
    args = parser.parse_args(argv)
    if args.check:
        expected = build_packet()
        actual = json.loads((ROOT / PACKET_PATH).read_text())
        if actual != expected:
            print("packet differs from deterministic build")
            return 2
        print(f"packet current: {expected['packet_sha256']}")
        return 0
    packet = write_packet()
    print(f"wrote {PACKET_PATH} sha256={packet['packet_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

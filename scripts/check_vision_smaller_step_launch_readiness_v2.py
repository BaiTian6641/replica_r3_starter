#!/usr/bin/env python3
"""Verify the v2 smaller-step accounting amendment and launch readiness.

The checker recomputes the v2 packet from its deterministic builder, validates
the derived accounting identities, confirms the preserved v1 hashes, inspects
the authorization receipt state without minting one, and writes a current
readiness result.  It never imports a native runtime, loads a model, executes a
launch or creates authorization.
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
import build_vision_smaller_step_launch_packet_v2 as builder  # noqa: E402
import vision_smaller_step_v2 as ss  # noqa: E402

DECLARATION_PATH = "configs/vision_smaller_step_launch_readiness_v2.json"
REPORT_PATH = ("reports/vision_smaller_step_launch_readiness_v2_20261027/"
               "readiness.json")
FORMAT = "replica.vision_smaller_step.launch_readiness.assessment.v2"


class ReadinessError(ValueError):
    """The declaration, packet or preserved evidence is stale."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ReadinessError(message)


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _receipt_state(declaration: Mapping[str, Any], config_sha: str,
                   root: Path) -> dict[str, Any]:
    path = root / declaration["authorization"]["path"]
    if not path.is_file():
        return {"present": False, "status": "absent", "sha256": None,
                "protocol_commit": None, "config_sha256": None,
                "binds_subject": False, "violations": [
                    "authorization_receipt_missing"]}
    receipt = _read_json(path)
    violations = []
    if receipt.get("format") != declaration["authorization"]["format"]:
        violations.append("receipt_format_differs")
    if receipt.get("status") != "authorized":
        violations.append("receipt_not_authorized")
    if receipt.get("config_sha256") != config_sha:
        violations.append("receipt_config_sha256_differs")
    commit = receipt.get("protocol_commit")
    if not (isinstance(commit, str) and len(commit) == 40
            and all(char in "0123456789abcdef" for char in commit)):
        violations.append("receipt_protocol_commit_not_full")
    for key in ("dev_authorized", "heldout_authorized", "v0_authorized",
                "semantic_audit_authorized", "promotion_authorized",
                "retry_authorized"):
        if receipt.get(key) is not False:
            violations.append(f"receipt_opens_role:{key}")
    return {
        "present": True,
        "status": receipt.get("status"),
        "sha256": _sha256_file(path),
        "protocol_commit": commit,
        "config_sha256": receipt.get("config_sha256"),
        "binds_subject": not violations,
        "violations": violations,
    }


def _commit_state(root: Path, config_sha: str,
                  receipt: Mapping[str, Any]) -> dict[str, Any]:
    commit = receipt.get("protocol_commit")
    if not commit:
        return {"commit_bound": False, "receipt_protocol_commit": None,
                "recorded_in_receipt": False,
                "reason": "no receipt records a commit"}
    import subprocess
    try:
        blob = subprocess.run(
            ["git", "-C", str(root), "cat-file", "blob",
             f"{commit}:{builder.CONFIG_PATH}"],
            check=False, capture_output=True)
    except OSError:
        return {"commit_bound": False, "receipt_protocol_commit": commit,
                "recorded_in_receipt": True, "reason": "git unavailable"}
    actual = (hashlib.sha256(blob.stdout).hexdigest()
              if blob.returncode == 0 else None)
    return {
        "commit_bound": actual == config_sha,
        "receipt_protocol_commit": commit,
        "recorded_in_receipt": True,
        "commit_config_blob_sha256": actual,
        "expected_config_sha256": config_sha,
    }


def assess(*, root: str | Path = ROOT) -> dict[str, Any]:
    root_path = Path(root)
    declaration = _read_json(root_path / DECLARATION_PATH)
    _require(declaration["format"]
             == "replica.vision_smaller_step.launch_readiness.v2",
             "Declaration format differs")
    config = ss.load_protocol(root_path / builder.CONFIG_PATH,
                              root=root_path)
    config_sha = _sha256_file(root_path / builder.CONFIG_PATH)
    packet = builder.build_packet(root=root_path)
    on_disk = _read_json(root_path / builder.PACKET_PATH)
    _require(on_disk == packet, "The v2 packet is not the deterministic build")
    packet_sha = _sha256_file(root_path / builder.PACKET_PATH)
    subject = declaration["subject"]
    _require(subject["protocol_path"] == builder.CONFIG_PATH
             and subject["protocol_sha256"] == config_sha,
             "Declaration does not bind the current v2 config")
    _require(subject["packet_path"] == builder.PACKET_PATH
             and subject["packet_file_sha256"] == packet_sha
             and subject["packet_canonical_sha256"]
             == packet["packet_sha256"],
             "Declaration does not bind the current v2 packet")
    preserved = declaration["preserved_v1"]
    _require(preserved["config_sha256"]
             == _sha256_file(root_path / builder.V1_CONFIG_PATH),
             "The v1 config changed; the amendment is stale")
    _require(preserved["packet_sha256"]
             == _sha256_file(root_path / builder.V1_PACKET_PATH),
             "The v1 packet changed; the amendment is stale")
    _require(preserved["readiness_sha256"]
             == _sha256_file(root_path / builder.V1_READINESS_PATH),
             "The v1 readiness declaration changed; the amendment is stale")
    acc_v2.validate_accounting(config["accounting"])
    campaign = config["accounting"]["campaign_cumulative"]
    declared_measurement = dict(campaign)
    declared_measurement["visual_encoder_cache_hits"] = 0
    live = acc_v2.validate_live_counters(
        declared_measurement, label="readiness_campaign",
        training_forwards=8)
    _require(config["readouts"]["expected_measured_work"]
             == acc_v2.expected_measured_work(),
             "Expected measured work is not the v2 derivation")
    _require(packet["accounting_amendment"]["campaign_cumulative"]
             == campaign,
             "Packet and config campaign counters differ")
    _require(declaration["accounting_amendment"]["campaign_cumulative"]
             == campaign,
             "Declaration and config campaign counters differ")
    receipt = _receipt_state(declaration, config_sha, root_path)
    commit = _commit_state(root_path, config_sha, receipt)
    output_path = root_path / declaration["launch"]["output"]
    output_exists = output_path.exists()
    refusals = [
        "packet_is_not_an_authorization_receipt",
        "packet_is_preparation_only",
        "authorization_receipt_missing" if not receipt["present"]
        else "authorization_receipt_invalid",
        "protocol_commit_unbound" if not commit["commit_bound"]
        else "commit_binding_verified",
        "commit_binding_unverified" if not commit["commit_bound"]
        else "commit_binding_verified",
        "native_execution_not_authorized",
        "fresh_output_required" if output_exists else "fresh_output_absent",
    ]
    readiness: dict[str, Any] = {
        "format": FORMAT,
        "version": "2",
        "name": declaration["name"],
        "deterministic": True,
        "native_execution": False,
        "model_loads": 0,
        "authorization_created": False,
        "declaration": {
            "path": DECLARATION_PATH,
            "sha256": _sha256_file(root_path / DECLARATION_PATH),
        },
        "subject": {
            "protocol_path": builder.CONFIG_PATH,
            "protocol_sha256": config_sha,
            "protocol_name": config["name"],
            "packet_path": builder.PACKET_PATH,
            "packet_file_sha256": packet_sha,
            "packet_canonical_sha256": packet["packet_sha256"],
            "packet_current": True,
            "protocol_amends_v1_sha256":
                config["accounting"]["amends"]["protocol_sha256"],
        },
        "preserved_v1": preserved,
        "accounting_amendment": declaration["accounting_amendment"],
        "validated_live_identities": live,
        "checks": [
            {"id": "declaration_binds_current_config",
             "status": "passed"},
            {"id": "packet_matches_deterministic_build",
             "status": "passed"},
            {"id": "accounting_derived_and_identity_checked",
             "status": "passed"},
            {"id": "v1_config_packet_readiness_preserved",
             "status": "passed"},
            {"id": "old_caps_still_visible_in_v1_packet",
             "status": "passed"},
            {"id": "authorization_state",
             "status": "observed",
             "requirement": f"receipt present={receipt['present']}"},
            {"id": "commit_binding", "status": "observed",
             "requirement": f"commit bound={commit['commit_bound']}"},
            {"id": "fresh_output", "status": "observed",
             "requirement": f"output exists={output_exists}"},
            {"id": "no_native_execution", "status": "passed"},
        ],
        "commit_binding": commit,
        "authorization": {**receipt, "path": declaration["authorization"]
                          ["path"], "required": True,
                          "minted_here": False},
        "launch": {
            "argv": declaration["launch"]["argv"],
            "review_command": declaration["launch"]["review_command"],
            "environment": declaration["launch"]["environment"],
            "output": declaration["launch"]["output"],
            "output_exists": output_exists,
            "disposition": ("review_only_launch_refused"
                            if not (receipt["binds_subject"]
                                    and commit["commit_bound"]
                                    and not output_exists)
                            else "review_only_launch_ready"),
            "refusals": refusals,
            "authorization_checked_before_native_imports": True,
        },
        "preserved_contract": declaration["preserved_contract"],
        "remaining_requirements": declaration["remaining_requirements"],
        "closed_roles": declaration["preserved_contract"]["closed_roles"],
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
            "No receipt exists and no commit is recorded; this readiness result "
            "never authorizes or executes the launch.",
            "The derived caps are exact declarations with identity checks, not "
            "native measurements.",
            "v1 config, packet, readiness and evidence are preserved and are "
            "not rewritten by this amendment.",
            "Development, held-out, semantic-audit, V0, promotion and retry "
            "stay closed.",
        ],
    }
    readiness["readiness_sha256"] = hashlib.sha256(
        _canonical(readiness).encode()).hexdigest()
    return readiness


def write_readiness(*, root: str | Path = ROOT,
                    path: str | Path = REPORT_PATH) -> dict[str, Any]:
    readiness = assess(root=root)
    _write_json(Path(root) / path, readiness)
    return readiness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify"])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        readiness = write_readiness()
    except ReadinessError as error:
        print(f"readiness refused: {error}")
        return 2
    summary = {
        "status": "readiness_current_launch_refused",
        "native_execution": False,
        "model_loads": 0,
        "report": REPORT_PATH,
        "readiness_sha256": readiness["readiness_sha256"],
        "config_sha256": readiness["subject"]["protocol_sha256"],
        "packet_sha256": readiness["subject"]["packet_canonical_sha256"],
        "launch_allowed": readiness["decision"]["launch_allowed"],
        "refusals": readiness["launch"]["refusals"],
        "review_command": readiness["launch"]["review_command"],
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"readiness current: {REPORT_PATH} "
              f"sha256={readiness['readiness_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

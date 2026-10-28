"""V2 smaller-step accounting, lineage and launch-refusal tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from replica_lab import vision_smaller_step_accounting_v2 as acc_v2  # noqa: E402
import build_vision_smaller_step_launch_packet_v2 as builder  # noqa: E402
import check_vision_smaller_step_launch_readiness_v2 as checker  # noqa: E402
import vision_smaller_step_v2 as ss  # noqa: E402


def _read(path: str | Path) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _sha256(path: str | Path) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def _link_inputs(root: Path) -> None:
    for name in ("configs", "data", "reports", "runs", "scripts", "src",
                 "checkpoints"):
        (root / name).symlink_to(ROOT / name, target_is_directory=True)


def test_v2_packet_is_deterministic_and_preserves_v1_lineage(tmp_path):
    _link_inputs(tmp_path)
    on_disk = _read(builder.PACKET_PATH)
    assert builder.build_packet(root=tmp_path) == on_disk
    assert on_disk["implementation"]["runner"]["path"].endswith(
        "vision_smaller_step_v2.py")
    assert on_disk["launch"]["argv"][1].endswith("vision_smaller_step_v2.py")
    assert on_disk["launch"]["review_command"][1].endswith(
        "vision_smaller_step_v2.py")

    v1 = _read(builder.V1_PACKET_PATH)
    runner = v1["implementation"]["runner"]
    assert runner["path"].endswith("vision_smaller_step_v1.py")
    assert _sha256(runner["path"]) == runner["sha256"]
    assert on_disk["preserved_v1"]["packet"]["sha256"] == _sha256(
        builder.V1_PACKET_PATH)


def test_v2_live_counter_identities_reject_one_unit_drift():
    config = ss.load_protocol(ROOT / builder.CONFIG_PATH)
    counters = dict(config["accounting"]["campaign_cumulative"])
    counters["visual_encoder_cache_hits"] = (
        config["accounting"]["expected_counts"]["campaign"]
        ["visual_encoder_cache_hits"])
    acc_v2.validate_live_counters(counters, label="test", training_forwards=8)
    for field in ("decoder_forward_calls", "visual_observation_calls",
                  "projector_passes", "processed_text_positions"):
        broken = dict(counters)
        broken[field] += 1
        with pytest.raises(acc_v2.AccountingDeclarationError):
            acc_v2.validate_live_counters(
                broken, label=f"test_{field}", training_forwards=8)


def test_v2_readiness_is_current_and_launch_remains_closed():
    readiness = checker.assess()
    assert readiness == _read(checker.REPORT_PATH)
    assert readiness["decision"]["launch_allowed"] is False
    assert readiness["decision"]["native_execution"] is False
    assert readiness["decision"]["model_loads"] == 0
    assert {"authorization_receipt_missing", "protocol_commit_unbound"} <= set(
        readiness["launch"]["refusals"])
    assert readiness["launch"]["review_command"][1].endswith(
        "vision_smaller_step_v2.py")


def test_v2_review_launch_refuses_missing_receipt_before_native(tmp_path):
    with pytest.raises(ss.ProtocolError, match="before model load"):
        ss.review_launch(
            ROOT / builder.CONFIG_PATH,
            authorization_path=tmp_path / "absent_authorization.json")


def test_v2_bound_review_emits_v2_commands_without_native_work(tmp_path):
    _link_inputs(tmp_path)
    config_path = ROOT / builder.CONFIG_PATH
    receipt = {
        "format": ss.AUTHORIZATION_FORMAT,
        "status": "authorized",
        "protocol_name": "vision_smaller_step_lr3e5_v2",
        "protocol_path": config_path.as_posix(),
        "protocol_commit": "a" * 40,
        "config_sha256": ss.protocol_sha256(config_path),
        "native_execution_authorized": True,
        "dev_authorized": False,
        "heldout_authorized": False,
        "v0_authorized": False,
        "semantic_audit_authorized": False,
        "promotion_authorized": False,
        "retry_authorized": False,
        "issued_at_utc": "2026-10-28T00:00:00+00:00",
    }
    authorization = tmp_path / "authorization.json"
    authorization.write_text(json.dumps(receipt), encoding="utf-8")
    review = ss.review_launch(
        config_path, authorization_path=authorization,
        root=tmp_path, verify_commit=False)
    assert review["native_execution"] is False
    assert review["model_loads"] == 0
    assert review["launch_command"][1].endswith("vision_smaller_step_v2.py")
    assert review["review_command"][1].endswith("vision_smaller_step_v2.py")

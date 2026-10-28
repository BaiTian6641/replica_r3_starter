"""Live artifact overlap and recovery tests."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab import artifact_publication as publication  # noqa: E402


def test_overlap_cap_refuses_replacement_and_allows_recovery(tmp_path):
    root = tmp_path / "artifacts"
    meter = publication.LiveByteMeter(root, cap_bytes=65_536)
    first = b"a" * 40_960
    publication.write_bytes(root, "payload.bin", first, meter=meter)
    assert (root / "payload.bin").read_bytes() == first

    with pytest.raises(publication.PublicationOverflow) as caught:
        publication.write_bytes(root, "payload.bin", b"b" * 40_960,
                                meter=meter)
    assert caught.value.measured_bytes == 81_920
    assert caught.value.cap_bytes == 65_536
    assert (root / "payload.bin").read_bytes() == first
    overflow = json.loads((root / "artifact_overflow.json").read_text())
    assert overflow["status"] == "overflow"
    assert overflow["measured_bytes_before_overflow_evidence"] == 81_920
    assert overflow["preserved"] is True
    assert overflow["deleted_artifacts"] == []
    assert overflow["projected_overlap"] is True
    assert not (root / ".payload.bin.tmp").exists()

    recovered = b"c" * 16_384
    publication.write_bytes(root, "payload.bin", recovered, meter=meter)
    assert (root / "payload.bin").read_bytes() == recovered
    assert publication.total_bytes(root) < 65_536
    assert meter.report()["peak_live_bytes"] == 81_920

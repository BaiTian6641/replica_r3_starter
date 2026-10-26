"""Deterministic tests for the bounded real-image tranche runner.

The tests never touch the network: every acquisition uses an injected fake
response object.  Decode/geometry/hash/dedup success paths use tiny Pillow
JPEGs and skip only when Pillow is not installed; protocol, cap, redirect and
fail-closed storage tests always run.  Phase-receipt regressions prove that
verification preserves seven-request acquisition evidence, writes separate
verification evidence with zero verify requests, and validates or explicitly
excludes incompatible acquisition history.
"""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import sys
import urllib.parse
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_real_image_tranche.py"
CONFIG = ROOT / "configs" / "real_image_tranche_v1.json"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tranche = _load(SCRIPT, "prepare_real_image_tranche")

PILLOW_AVAILABLE = importlib.util.find_spec("PIL") is not None
requires_pillow = pytest.mark.skipif(
    not PILLOW_AVAILABLE,
    reason="Pillow is required to generate/verify JPEG fixtures: install Pillow",
)


# ---------------------------------------------------------------------------
# protocol/config fixtures
# ---------------------------------------------------------------------------


def _real_protocol() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _all_assets(protocol: dict) -> list[dict]:
    return [asset for source in protocol["sources"] for asset in source["assets"]]


def _write_config(root: Path, mutate=None) -> Path:
    protocol = copy.deepcopy(_real_protocol())
    if mutate is not None:
        mutate(protocol)
    path = root / "configs" / "real_image_tranche_v1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return path


def _plan(config_path: Path):
    return tranche.load_plan(config_path)


# ---------------------------------------------------------------------------
# fake HTTP boundary
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(
        self,
        url: str,
        data: bytes,
        *,
        content_type: str | None = "image/jpeg",
        content_length: int | str | None = "auto",
        chunk_size: int = 97,
        final_url: str | None = None,
    ):
        self._url = final_url or url
        self._data = data
        self._position = 0
        self._chunk_size = max(1, int(chunk_size))
        self.closed = False
        self.headers: dict[str, str] = {}
        if content_type is not None:
            self.headers["Content-Type"] = content_type
        if content_length == "auto":
            self.headers["Content-Length"] = str(len(data))
        elif content_length is not None:
            self.headers["Content-Length"] = str(content_length)

    def read(self, size: int = -1) -> bytes:
        if self._position >= len(self._data):
            return b""
        if size is None or size < 0:
            size = len(self._data) - self._position
        size = min(size, self._chunk_size)
        chunk = self._data[self._position : self._position + size]
        self._position += len(chunk)
        return chunk

    def geturl(self) -> str:
        return self._url

    def close(self) -> None:
        self.closed = True


def _opener_for(specs: dict[str, dict]):
    calls: list[dict] = []

    def opener(url: str, timeout: float):
        calls.append({"url": url, "timeout": timeout})
        spec = specs[url]
        if isinstance(spec, BaseException):
            raise spec
        return FakeResponse(url, **spec)

    return opener, calls


def _specs_for_plan(plan, data_by_asset: dict[str, bytes], **options) -> dict:
    return {
        record["thumb_url"]: {
            "data": data_by_asset[record["asset_id"]],
            **options,
        }
        for record in plan.assets
    }


def _jpeg(width: int, height: int, color: tuple[int, int, int]) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(
        buffer, format="JPEG", quality=90
    )
    return buffer.getvalue()


def _png(width: int, height: int, color: tuple[int, int, int]) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _valid_jpeg_map(plan) -> dict[str, bytes]:
    payloads = {}
    for index, record in enumerate(plan.assets):
        height = round(
            record["original_height"]
            * plan.derivative_width
            / record["original_width"]
        )
        payloads[record["asset_id"]] = _jpeg(
            plan.derivative_width,
            height,
            ((index * 37) % 256, (index * 71 + 40) % 256, (index * 113 + 80) % 256),
        )
    return payloads


def _download(tmp_path: Path, config_path: Path, specs: dict, *, assets_name="assets"):
    opener, calls = _opener_for(specs)
    report = tranche.download(
        config_path,
        tmp_path / assets_name,
        tmp_path / "reports",
        opener=opener,
    )
    return report, calls, tmp_path / assets_name, tmp_path / "reports"


def _assert_no_admitted_set(assets_dir: Path, report_dir: Path) -> None:
    assert not assets_dir.exists(), "a failed run must not leave a partial asset set"
    assert not list(
        assets_dir.parent.glob(f".{assets_dir.name}.stage-*")
    ), "a failed run must remove staged bytes"
    assert not (report_dir / tranche.MANIFEST_NAME).exists(), (
        "a failed run must not write an admission manifest"
    )
    assert (report_dir / tranche.REPORT_NAME).is_file(), (
        "a failed run must preserve a failure report"
    )
    assert (report_dir / tranche.ACQUISITION_NAME).is_file(), (
        "a failed download must preserve an acquisition receipt"
    )
    assert not (report_dir / tranche.VERIFICATION_NAME).exists(), (
        "a failed download must not write a verification receipt"
    )


# ---------------------------------------------------------------------------
# frozen protocol and pre-network validation
# ---------------------------------------------------------------------------


def test_frozen_protocol_plan_and_split_assignment() -> None:
    plan = _plan(CONFIG)
    assert plan.exact_assets == 7
    assert plan.exact_sources == 6
    assert len(plan.assets) == 7
    assert plan.asset_counts == {"train": 4, "development": 1, "audit": 2}
    assert plan.group_splits == {
        "nasa_apollo": "train",
        "loc_fsa_owi": "audit",
        "rijksmuseum": "train",
        "met_open_access": "audit",
        "usgs": "train",
        "us_national_archives": "development",
    }
    assert plan.expected_group_splits == plan.group_splits
    assert plan.expected_asset_counts == plan.asset_counts
    assert plan.derivative_width == 960
    caps = plan.protocol["caps"]
    salt = plan.protocol["split_assignment"]["salt"]
    for group, expected in plan.expected_group_splits.items():
        assert tranche.assign_split(group, salt, caps) == expected
    assert len({record["asset_id"] for record in plan.assets}) == 7
    assert len({record["thumb_url"] for record in plan.assets}) == 7
    assert len({record["source_page"] for record in plan.assets}) == 7


def _invalid_mutations():
    def drop_asset(protocol):
        protocol["sources"][0]["assets"].pop()

    def duplicate_asset_id(protocol):
        assets = _all_assets(protocol)
        assets[1]["asset_id"] = assets[0]["asset_id"]

    def duplicate_source_page(protocol):
        assets = _all_assets(protocol)
        assets[1]["source_page"] = assets[0]["source_page"]

    def duplicate_thumb(protocol):
        assets = _all_assets(protocol)
        assets[1]["thumb_url"] = assets[0]["thumb_url"]

    def missing_rights(protocol):
        del protocol["sources"][0]["assets"][0]["rights_basis"]

    def empty_license_name(protocol):
        protocol["sources"][0]["assets"][0]["license_short_name"] = ""

    def bad_license(protocol):
        protocol["sources"][0]["assets"][0]["license"] = "cc-by-nc"

    def bad_revision(protocol):
        protocol["sources"][0]["assets"][0]["revision_id"] = "main"

    def bad_original_hash(protocol):
        protocol["sources"][0]["assets"][0]["original_sha1"] = "deadbeef"

    def source_jurisdiction(protocol):
        protocol["sources"][0]["jurisdiction"] = "DE"

    def top_jurisdiction(protocol):
        protocol["jurisdiction"]["intended"] = "CA"

    def evil_host(protocol):
        asset = protocol["sources"][0]["assets"][0]
        asset["thumb_url"] = asset["thumb_url"].replace(
            "thumb.wikimedia.org", "evil.example"
        )

    def plain_http(protocol):
        asset = protocol["sources"][0]["assets"][0]
        asset["thumb_url"] = asset["thumb_url"].replace("https://", "http://", 1)

    def wrong_width(protocol):
        asset = protocol["sources"][0]["assets"][0]
        asset["thumb_url"] = asset["thumb_url"].replace("/960px-", "/800px-")

    def split_mismatch(protocol):
        protocol["split_assignment"]["expected_group_splits"]["usgs"] = "audit"

    def count_mismatch(protocol):
        protocol["split_assignment"]["expected_asset_counts"]["train"] = 3

    def benchmark_id(protocol):
        protocol["sources"][0]["assets"][0]["asset_id"] = "apollo_mmlu"

    def bulk_download(protocol):
        protocol["acquisition"]["bulk_download"] = True

    def request_budget(protocol):
        protocol["acquisition"]["max_requests"] = 6

    def duplicate_source_id(protocol):
        protocol["sources"][1]["source_id"] = protocol["sources"][0]["source_id"]

    return {
        "asset_count": drop_asset,
        "duplicate_asset_id": duplicate_asset_id,
        "duplicate_source_page": duplicate_source_page,
        "duplicate_thumb_url": duplicate_thumb,
        "missing_rights_basis": missing_rights,
        "empty_license_short_name": empty_license_name,
        "disallowed_license": bad_license,
        "unpinned_revision": bad_revision,
        "bad_original_sha1": bad_original_hash,
        "source_jurisdiction": source_jurisdiction,
        "top_jurisdiction": top_jurisdiction,
        "evil_thumb_host": evil_host,
        "plain_http_thumb": plain_http,
        "wrong_derivative_width": wrong_width,
        "expected_split_mismatch": split_mismatch,
        "expected_count_mismatch": count_mismatch,
        "benchmark_identifier": benchmark_id,
        "bulk_download": bulk_download,
        "request_budget_too_small": request_budget,
        "duplicate_source_id": duplicate_source_id,
    }


def test_entire_protocol_is_validated_before_any_request(tmp_path: Path) -> None:
    for name, mutate in _invalid_mutations().items():
        case_root = tmp_path / name
        config_path = _write_config(case_root, mutate)
        calls: list[dict] = []

        def opener(url: str, timeout: float):
            calls.append({"url": url, "timeout": timeout})
            raise AssertionError("network must not be touched before validation")

        report = tranche.download(
            config_path,
            case_root / "assets",
            case_root / "reports",
            opener=opener,
        )
        assert report["status"] == "refused", name
        assert report["failures"][0]["reason"] == "protocol_invalid", name
        assert report["requests"]["attempted"] == 0, name
        assert report["network_used"] is False, name
        assert calls == [], name
        _assert_no_admitted_set(case_root / "assets", case_root / "reports")


def test_frozen_nullable_rights_fields_are_accepted() -> None:
    plan = _plan(CONFIG)
    for record in plan.assets:
        if record["license"] == "pd":
            assert record["license_url"] is None
        assert isinstance(record["restrictions"], str)
        assert record["rights_basis"]
        assert record["institutional_source"].startswith("https://")


# ---------------------------------------------------------------------------
# bounded fetch failures (no Pillow required)
# ---------------------------------------------------------------------------


def test_destination_with_content_is_never_overwritten(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    plan = _plan(config_path)
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "stray.jpg").write_bytes(b"not ours")
    calls: list[dict] = []

    def opener(url: str, timeout: float):
        calls.append({"url": url, "timeout": timeout})
        raise AssertionError("must not fetch into a non-empty destination")

    report = tranche.download(
        config_path, assets_dir, tmp_path / "reports", opener=opener
    )
    assert report["status"] == "refused"
    assert report["failures"][0]["reason"] == "assets_dir_not_usable"
    assert report["network_used"] is False
    assert calls == []
    assert (assets_dir / "stray.jpg").read_bytes() == b"not ours"
    assert not (tmp_path / "reports" / tranche.MANIFEST_NAME).exists()
    assert (tmp_path / "reports" / tranche.REPORT_NAME).is_file()
    assert (tmp_path / "reports" / tranche.ACQUISITION_NAME).is_file()
    assert not (tmp_path / "reports" / tranche.VERIFICATION_NAME).exists()
    receipt = json.loads(
        (tmp_path / "reports" / tranche.ACQUISITION_NAME).read_text()
    )
    assert receipt["format"] == tranche.ACQUISITION_FORMAT
    assert receipt["status"] == "refused"
    assert receipt["manifest"] is None
    assert plan.exact_assets == 7  # plan really was valid before the destination stop


def test_per_asset_and_total_byte_caps_fail_closed(tmp_path: Path) -> None:
    def small_asset(protocol):
        protocol["acquisition"]["max_asset_bytes"] = 64
        protocol["acquisition"]["max_total_bytes"] = 1024

    config_path = _write_config(tmp_path / "asset", small_asset)
    plan = _plan(config_path)
    specs = _specs_for_plan(
        plan,
        {record["asset_id"]: b"x" * 100 for record in plan.assets},
        content_length="auto",
    )
    report, calls, assets_dir, report_dir = _download(
        tmp_path / "asset", config_path, specs
    )
    assert report["status"] == "refused"
    assert any(
        failure["reason"] == "asset_byte_cap_exceeded"
        for failure in report["failures"]
    )
    assert report["bytes"]["received"] == 0
    assert report["requests"]["attempted"] == 7
    assert report["training_admitted"] is False
    assert report["non_claims"]["perceptual_dedup_claimed"] is False
    assert report["non_claims"]["model_quality_evaluated"] is False
    assert calls
    _assert_no_admitted_set(assets_dir, report_dir)

    def small_total(protocol):
        protocol["acquisition"]["max_asset_bytes"] = 64
        protocol["acquisition"]["max_total_bytes"] = 128

    config_path = _write_config(tmp_path / "total", small_total)
    plan = _plan(config_path)
    specs = _specs_for_plan(
        plan,
        {record["asset_id"]: b"y" * 64 for record in plan.assets},
        content_length=None,
    )
    report, calls, assets_dir, report_dir = _download(
        tmp_path / "total", config_path, specs
    )
    assert report["status"] == "refused"
    assert any(
        failure["reason"] == "total_byte_cap_exceeded"
        for failure in report["failures"]
    )
    assert report["requests"]["attempted"] == 2
    assert report["bytes"]["received"] == 128
    _assert_no_admitted_set(assets_dir, report_dir)

    def stream_cap(protocol):
        protocol["acquisition"]["max_asset_bytes"] = 64
        protocol["acquisition"]["max_total_bytes"] = 1024

    config_path = _write_config(tmp_path / "stream", stream_cap)
    plan = _plan(config_path)
    specs = _specs_for_plan(
        plan,
        {record["asset_id"]: b"z" * 200 for record in plan.assets},
        content_length=None,
        chunk_size=50,
    )
    report, calls, assets_dir, report_dir = _download(
        tmp_path / "stream", config_path, specs
    )
    assert report["status"] == "refused"
    assert any(
        failure["reason"] == "asset_byte_cap_exceeded"
        for failure in report["failures"]
    )
    assert report["requests"]["attempted"] == 7
    assert all(call["timeout"] == 30 for call in calls)
    _assert_no_admitted_set(assets_dir, report_dir)


def test_redirects_are_rechecked_before_body_read(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    plan = _plan(config_path)
    specs = {}
    for record in plan.assets:
        parsed = urllib.parse.urlsplit(record["thumb_url"])
        redirect_url = urllib.parse.urlunsplit(
            ("https", "upload.wikimedia.org", parsed.path, "", "")
        )
        specs[record["thumb_url"]] = {
            "data": b"must not be read",
            "final_url": redirect_url,
        }
    report, calls, assets_dir, report_dir = _download(tmp_path, config_path, specs)
    assert report["status"] == "refused"
    reasons = {failure["reason"] for failure in report["failures"]}
    assert "redirect_rejected" in reasons
    assert "fetch_error" not in reasons
    assert report["bytes"]["received"] == 0
    assert report["requests"]["attempted"] == len(plan.assets)
    _assert_no_admitted_set(assets_dir, report_dir)


def test_content_type_and_length_boundaries_fail_closed(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "content_type")
    plan = _plan(config_path)
    specs = _specs_for_plan(
        plan,
        {record["asset_id"]: b"<html>not an image</html>" for record in plan.assets},
        content_type="text/html",
    )
    report, calls, assets_dir, report_dir = _download(
        tmp_path / "content_type", config_path, specs
    )
    assert report["status"] == "refused"
    reasons = {failure["reason"] for failure in report["failures"]}
    assert "wrong_content_type" in reasons
    assert report["bytes"]["received"] == 0
    _assert_no_admitted_set(assets_dir, report_dir)

    config_path = _write_config(tmp_path / "length")
    plan = _plan(config_path)
    specs = _specs_for_plan(
        plan,
        {record["asset_id"]: b"short" for record in plan.assets},
        content_length=100,
    )
    report, calls, assets_dir, report_dir = _download(
        tmp_path / "length", config_path, specs
    )
    assert report["status"] == "refused"
    reasons = {failure["reason"] for failure in report["failures"]}
    assert "content_length_mismatch" in reasons
    assert report["bytes"]["received"] == 5 * len(plan.assets)
    _assert_no_admitted_set(assets_dir, report_dir)


@pytest.mark.skipif(PILLOW_AVAILABLE, reason="exercises the Pillow-absent boundary")
def test_missing_pillow_refuses_without_partial_admission(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    plan = _plan(config_path)
    specs = _specs_for_plan(
        plan,
        {record["asset_id"]: b"not decodable" for record in plan.assets},
    )
    report, calls, assets_dir, report_dir = _download(tmp_path, config_path, specs)
    assert report["status"] == "refused"
    assert any(
        failure["reason"] == "pillow_unavailable"
        for failure in report["failures"]
    )
    assert report["requests"]["attempted"] == 7
    _assert_no_admitted_set(assets_dir, report_dir)


# ---------------------------------------------------------------------------
# Pillow decode/geometry/hash/dedup/all-or-nothing admission
# ---------------------------------------------------------------------------


@requires_pillow
def test_download_then_verify_preserves_acquisition_evidence(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    plan = _plan(config_path)
    payloads = _valid_jpeg_map(plan)
    specs = _specs_for_plan(plan, payloads, chunk_size=53)
    report, calls, assets_dir, report_dir = _download(tmp_path, config_path, specs)

    assert report["status"] == "admitted"
    assert report["failures"] == []
    assert report["admitted_count"] == 7
    assert report["requests"]["attempted"] == 7
    assert report["requests"]["succeeded"] == 7
    assert report["network_used"] is True
    assert report["training_admitted"] is False
    assert report["non_claims"]["perceptual_dedup_claimed"] is False
    assert report["checks"]["all_checks_passed"] is True
    assert all(report["checks"][name] for name in report["checks"] if name.startswith("all_"))

    expected_names = {f"{record['asset_id']}.jpg" for record in plan.assets}
    assert {path.name for path in assets_dir.iterdir()} == expected_names
    assert not list(assets_dir.parent.glob(f".{assets_dir.name}.stage-*"))
    assert (report_dir / tranche.MANIFEST_NAME).is_file()
    assert (report_dir / tranche.REPORT_NAME).is_file()
    acquisition_path = report_dir / tranche.ACQUISITION_NAME
    assert acquisition_path.is_file()
    assert not (report_dir / tranche.VERIFICATION_NAME).exists()
    assert report["evidence"][tranche.ACQUISITION_NAME]["written"] is True
    acquisition_before = acquisition_path.read_bytes()
    acquisition = json.loads(acquisition_before)
    assert acquisition["format"] == tranche.ACQUISITION_FORMAT
    assert acquisition["mode"] == "download"
    assert acquisition["status"] == "admitted"
    assert acquisition["network_used"] is True
    assert acquisition["requests"]["attempted"] == 7
    assert acquisition["requests"]["succeeded"] == 7
    assert acquisition["requests"]["failed"] == 0
    assert acquisition["admitted_count"] == acquisition["asset_count"] == 7
    assert acquisition["manifest"]["asset_count"] == 7
    assert acquisition["manifest"]["total_bytes"] == acquisition["bytes"]["admitted"]
    assert len(acquisition["manifest"]["asset_hashes"]) == 7
    assert acquisition["failures"] == []
    assert acquisition["training_admitted"] is False

    manifest = json.loads((report_dir / tranche.MANIFEST_NAME).read_text())
    assert manifest["format"] == tranche.MANIFEST_FORMAT
    assert manifest["status"] == tranche.MANIFEST_STATUS
    assert manifest["protocol"]["sha256"] == tranche.sha256_file(config_path)
    assert manifest["implementation"]["sha256"] == tranche._implementation_info()["sha256"]
    assert manifest["asset_count"] == 7
    assert manifest["source_count"] == 6
    assert manifest["training_admitted"] is False
    assert manifest["perceptual_dedup_claimed"] is False
    assert manifest["splits"] == report["splits"]
    assert manifest["splits"]["train"]["assets"] == 4
    assert manifest["splits"]["development"]["assets"] == 1
    assert manifest["splits"]["audit"]["assets"] == 2
    assert manifest["total_bytes"] == sum(
        entry["derivative"]["bytes"] for entry in manifest["assets"]
    )
    assert report["storage"]["manifest_bytes"] == len(
        (report_dir / tranche.MANIFEST_NAME).read_bytes()
    )
    assert report["storage"]["total_stored_bytes"] == (
        manifest["total_bytes"] + report["storage"]["manifest_bytes"]
    )
    for entry in manifest["assets"]:
        assert entry["source"]["rights_basis"]
        assert entry["source"]["institutional_source"].startswith("https://")
        assert entry["source"]["revision_id"] > 0
        assert tranche.SHA1_PATTERN.fullmatch(entry["source"]["original_sha1"])
        assert entry["derivative"]["mime"] == "image/jpeg"
        assert entry["derivative"]["format"] == "JPEG"
        assert entry["derivative"]["width"] == 960
        assert entry["derivative"]["height"] > 0
        assert tranche.SHA256_PATTERN.fullmatch(entry["derivative"]["sha256"])
        assert entry["derivative"]["bytes"] == len(
            (assets_dir / entry["filename"]).read_bytes()
        )
        assert all(entry["checks"][name] for name in tranche.ASSET_CHECK_NAMES)
    portrait = next(
        entry
        for entry in manifest["assets"]
        if entry["asset_id"] == "loc_migrant_mother"
    )
    assert portrait["derivative"]["width"] == 960
    assert portrait["derivative"]["height"] > 960
    assert portrait["derivative"]["max_dimension"] == portrait["derivative"]["height"]

    manifest_before = (report_dir / tranche.MANIFEST_NAME).read_bytes()
    verified = tranche.verify(config_path, assets_dir, report_dir)
    assert verified["status"] == "verified"
    assert verified["verified"] is True
    assert verified["failures"] == []
    assert verified["network_used"] is False
    assert verified["requests"]["attempted"] == 0
    assert verified["requests"]["succeeded"] == 0
    assert verified["requests"]["failed"] == 0
    assert verified["checks"]["all_checks_passed"] is True
    assert verified["verification"]["manifest_sha256"] == tranche.sha256_bytes(
        manifest_before
    )
    assert verified["verification"]["acquisition_receipt"] == "included"
    assert verified["acquisition"]["requests"]["attempted"] == 7
    assert verified["acquisition"]["requests"]["succeeded"] == 7
    assert verified["acquisition"]["network_used"] is True
    assert verified["acquisition"]["status"] == "admitted"
    assert verified["acquisition"]["failures"] == []
    assert verified["acquisition"]["started_at_utc"]
    assert verified["acquisition"]["completed_at_utc"]
    assert isinstance(verified["acquisition"]["wall_time_seconds"], (int, float))
    assert acquisition_path.read_bytes() == acquisition_before, (
        "verify must not rewrite the acquisition receipt"
    )
    verification_path = report_dir / tranche.VERIFICATION_NAME
    assert verification_path.is_file()
    assert verified["evidence"][tranche.VERIFICATION_NAME]["written"] is True
    verification = json.loads(verification_path.read_text())
    assert verification["format"] == tranche.VERIFICATION_FORMAT
    assert verification["mode"] == "verify"
    assert verification["status"] == "verified"
    assert verification["network_used"] is False
    assert verification["requests"]["attempted"] == 0
    assert verification["acquisition"]["requests"]["attempted"] == 7
    assert verification["acquisition"]["network_used"] is True
    assert verification["manifest"]["sha256"] == tranche.sha256_bytes(manifest_before)
    assert (report_dir / tranche.MANIFEST_NAME).read_bytes() == manifest_before


@requires_pillow
def test_download_is_all_or_nothing_on_bad_geometry_and_format(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "geometry")
    plan = _plan(config_path)
    payloads = _valid_jpeg_map(plan)
    victim = plan.assets[0]["asset_id"]
    payloads[victim] = _jpeg(960, 200, (1, 2, 3))
    specs = _specs_for_plan(plan, payloads)
    report, calls, assets_dir, report_dir = _download(
        tmp_path / "geometry", config_path, specs
    )
    assert report["status"] == "refused"
    assert any(failure["reason"] == "wrong_geometry" for failure in report["failures"])
    _assert_no_admitted_set(assets_dir, report_dir)

    config_path = _write_config(tmp_path / "format")
    plan = _plan(config_path)
    payloads = _valid_jpeg_map(plan)
    victim = plan.assets[0]
    expected_height = round(
        victim["original_height"] * plan.derivative_width / victim["original_width"]
    )
    payloads[victim["asset_id"]] = _png(
        plan.derivative_width, expected_height, (9, 9, 9)
    )
    specs = _specs_for_plan(plan, payloads)
    report, calls, assets_dir, report_dir = _download(
        tmp_path / "format", config_path, specs
    )
    assert report["status"] == "refused"
    assert any(failure["reason"] == "wrong_format" for failure in report["failures"])
    _assert_no_admitted_set(assets_dir, report_dir)


@requires_pillow
def test_truncated_jpeg_is_rejected_without_partial_admission(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    plan = _plan(config_path)
    payloads = _valid_jpeg_map(plan)
    victim = plan.assets[0]["asset_id"]
    payloads[victim] = payloads[victim][: len(payloads[victim]) // 2]
    specs = _specs_for_plan(plan, payloads)
    report, calls, assets_dir, report_dir = _download(tmp_path, config_path, specs)
    assert report["status"] == "refused"
    reasons = {failure["reason"] for failure in report["failures"]}
    assert reasons & {"truncated_image", "wrong_format"}
    _assert_no_admitted_set(assets_dir, report_dir)


@requires_pillow
def test_exact_duplicate_bytes_are_rejected_globally(tmp_path: Path) -> None:
    def same_geometry(protocol):
        assets = _all_assets(protocol)
        assets[1]["original_width"] = assets[0]["original_width"]
        assets[1]["original_height"] = assets[0]["original_height"]

    config_path = _write_config(tmp_path, same_geometry)
    plan = _plan(config_path)
    payloads = _valid_jpeg_map(plan)
    first, second = plan.assets[0], plan.assets[1]
    payloads[second["asset_id"]] = payloads[first["asset_id"]]
    specs = _specs_for_plan(plan, payloads)
    report, calls, assets_dir, report_dir = _download(tmp_path, config_path, specs)
    assert report["status"] == "refused"
    assert any(
        failure["reason"] == "duplicate_exact_sha256"
        for failure in report["failures"]
    )
    _assert_no_admitted_set(assets_dir, report_dir)


@requires_pillow
def test_verification_rejects_asset_and_manifest_drift(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    plan = _plan(config_path)
    payloads = _valid_jpeg_map(plan)
    specs = _specs_for_plan(plan, payloads)
    report, calls, assets_dir, report_dir = _download(tmp_path, config_path, specs)
    assert report["status"] == "admitted"

    manifest_path = report_dir / tranche.MANIFEST_NAME
    manifest_before = manifest_path.read_bytes()
    manifest = json.loads(manifest_before)
    tampered = copy.deepcopy(manifest)
    tampered["assets"][0]["source"]["rights_basis"] = "fabricated rights"
    manifest_path.write_text(json.dumps(tampered, indent=2) + "\n", encoding="utf-8")
    drifted = tranche.verify(config_path, assets_dir, report_dir)
    assert drifted["status"] == "verification_failed"
    assert any(
        failure["reason"] == "manifest_source_drift"
        for failure in drifted["failures"]
    )
    assert drifted["network_used"] is False
    assert drifted["checks"]["all_checks_passed"] is False

    manifest_path.write_bytes(manifest_before)
    target = assets_dir / manifest["assets"][0]["filename"]
    original = target.read_bytes()
    target.write_bytes(original[:-1] + bytes([original[-1] ^ 0x01]))
    drifted = tranche.verify(config_path, assets_dir, report_dir)
    assert drifted["status"] == "verification_failed"
    reasons = {failure["reason"] for failure in drifted["failures"]}
    assert reasons & {
        "sha256_mismatch",
        "byte_length_mismatch",
        "derivative_metadata_drift",
        "truncated_image",
        "check_failed",
    }
    assert drifted["network_used"] is False
    assert manifest_path.read_bytes() == manifest_before


@requires_pillow
def test_acquisition_history_is_validated_fail_closed_or_excluded(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    plan = _plan(config_path)
    payloads = _valid_jpeg_map(plan)
    report, _, assets_dir, report_dir = _download(
        tmp_path / "success", config_path, _specs_for_plan(plan, payloads)
    )
    assert report["status"] == "admitted"
    acquisition_path = report_dir / tranche.ACQUISITION_NAME
    acquisition_before = acquisition_path.read_bytes()
    acquisition = json.loads(acquisition_before)

    tampered = copy.deepcopy(acquisition)
    tampered["requests"]["attempted"] = 6  # one fewer than the seven admitted assets
    acquisition_path.write_text(tranche.serialize_json(tampered), encoding="utf-8")
    tampered_bytes = acquisition_path.read_bytes()
    failed = tranche.verify(config_path, assets_dir, report_dir)
    assert failed["status"] == "verification_failed"
    assert any(
        failure["reason"] == "acquisition_incompatible"
        for failure in failed["failures"]
    )
    assert failed["acquisition"] is None
    assert failed["verification"]["acquisition_receipt"] == "incompatible"
    assert failed["requests"]["attempted"] == 0
    assert failed["network_used"] is False
    assert len(failed["asset_results"]) == 7
    assert all(
        all(entry["checks"].values()) for entry in failed["asset_results"]
    ), "offline asset/hash QA must remain authoritative"
    assert acquisition_path.read_bytes() == tampered_bytes, (
        "verify must not rewrite the incompatible acquisition receipt"
    )
    assert (report_dir / tranche.VERIFICATION_NAME).is_file()
    failed_receipt = json.loads(
        (report_dir / tranche.VERIFICATION_NAME).read_text()
    )
    assert failed_receipt["status"] == "verification_failed"
    assert failed_receipt["acquisition"] is None
    assert failed_receipt["requests"]["attempted"] == 0

    # A structurally valid but refused acquisition attempt is excluded from
    # the authoritative evidence and reported with an explicit warning.
    acquisition_path.write_bytes(acquisition_before)
    failure_specs = _specs_for_plan(
        plan,
        {record["asset_id"]: b"<html>not a jpeg</html>" for record in plan.assets},
    )
    refused_report, _, refused_assets, refused_reports = _download(
        tmp_path / "failure", config_path, failure_specs
    )
    assert refused_report["status"] == "refused"
    refused_path = refused_reports / tranche.ACQUISITION_NAME
    refused = json.loads(refused_path.read_text())
    assert refused["status"] == "refused"
    assert refused["manifest"] is None
    assert not refused_assets.exists()
    acquisition_path.write_bytes(refused_path.read_bytes())
    refused_bytes = acquisition_path.read_bytes()

    warned = tranche.verify(config_path, assets_dir, report_dir)
    assert warned["status"] == "verified"
    assert warned["acquisition"] is None
    assert warned["verification"]["acquisition_receipt"] == "not_admitted"
    assert "acquisition_receipt_not_admitted" in warned["warnings"]
    assert warned["requests"]["attempted"] == 0
    assert warned["network_used"] is False
    assert acquisition_path.read_bytes() == refused_bytes


@requires_pillow
def test_report_serialization_is_deterministic_except_timing(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    plan = _plan(config_path)
    payloads = _valid_jpeg_map(plan)

    first_report, _, first_assets, first_reports = _download(
        tmp_path / "first", config_path, _specs_for_plan(plan, payloads)
    )
    second_report, _, second_assets, second_reports = _download(
        tmp_path / "second", config_path, _specs_for_plan(plan, payloads)
    )
    assert first_report["status"] == second_report["status"] == "admitted"

    first_manifest = (first_reports / tranche.MANIFEST_NAME).read_bytes()
    second_manifest = (second_reports / tranche.MANIFEST_NAME).read_bytes()
    assert first_manifest == second_manifest

    def deterministic(report: dict) -> dict:
        normalized = copy.deepcopy(report)
        for field in ("started_at_utc", "completed_at_utc", "wall_time_seconds"):
            normalized.pop(field)
        for field in ("assets_dir", "report_dir", "report_path"):
            normalized["storage"].pop(field, None)
        normalized["verification"].pop("manifest_path", None)
        normalized.pop("evidence", None)
        return normalized

    assert deterministic(first_report) == deterministic(second_report)
    assert tranche.serialize_json(deterministic(first_report)) == tranche.serialize_json(
        deterministic(second_report)
    )

    def deterministic_receipt(receipt: dict) -> dict:
        normalized = copy.deepcopy(receipt)
        for field in ("started_at_utc", "completed_at_utc", "wall_time_seconds"):
            normalized.pop(field)
        return normalized

    first_acquisition = json.loads(
        (first_reports / tranche.ACQUISITION_NAME).read_text()
    )
    second_acquisition = json.loads(
        (second_reports / tranche.ACQUISITION_NAME).read_text()
    )
    assert deterministic_receipt(first_acquisition) == deterministic_receipt(
        second_acquisition
    )

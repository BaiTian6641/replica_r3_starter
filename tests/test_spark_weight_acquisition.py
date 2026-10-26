"""Offline tests for bounded Spark-X2.6 weight acquisition.

Every test uses synthetic, bounded byte streams and temporary directories. No
test contacts Hugging Face, imports torch/transformers/safetensors, executes
remote code or writes into the repository's real ``checkpoints/`` directory.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/acquire_spark_weights.py"
SPEC = importlib.util.spec_from_file_location("acquire_spark_weights", SCRIPT)
acq = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(acq)

REAL_PLAN = ROOT / "configs/spark_weight_acquisition_v1.json"


# ---------------------------------------------------------------------------
# Synthetic bounded transport


class FakeResponse:
    def __init__(self, body: bytes, *, start: int, total: int,
                 declared_length: int | None = None, status: int | None = None):
        self._body = body
        self._pos = 0
        if start:
            self.status = status or 206
            self.headers = {
                "Content-Length": str(len(body)),
                "Content-Range": f"bytes {start}-{start + len(body) - 1}/{total}",
                "Content-Encoding": "identity",
            }
        else:
            self.status = status or 200
            self.headers = {
                "Content-Length": str(len(body) if declared_length is None
                                      else declared_length),
                "Content-Encoding": "identity",
            }

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = len(self._body) - self._pos
        chunk = self._body[self._pos:self._pos + size]
        self._pos += len(chunk)
        return chunk

    def close(self) -> None:
        pass


class FakeTransport:
    """Serves pinned bytes by file name, honouring resume ranges."""

    def __init__(self, payloads, *, declared_lengths=None):
        self.payloads = dict(payloads)
        self.declared_lengths = dict(declared_lengths or {})
        self.requests = 0
        self.calls: list[tuple[str, int]] = []

    def open(self, url: str, *, start: int = 0):
        self.requests += 1
        filename = url.rsplit("/", 1)[-1]
        self.calls.append((filename, start))
        if filename not in self.payloads:
            raise AssertionError(f"unexpected URL: {url}")
        body = self.payloads[filename]
        total = len(body)
        return FakeResponse(
            body[start:], start=start, total=total,
            declared_length=self.declared_lengths.get(filename))


class ExplodingTransport:
    def __init__(self):
        self.requests = 0

    def open(self, url: str, *, start: int = 0):
        self.requests += 1
        raise AssertionError(f"network access is forbidden here: {url}")


class FreeBytes:
    def __init__(self, value: int = 10 ** 9):
        self.value = value

    def __call__(self, path: Path) -> int:
        return self.value


# ---------------------------------------------------------------------------
# Fixture construction


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def default_source_spec(**overrides) -> dict:
    spec = {
        "role": "official_fp8_adaptation_base",
        "repo_id": "Example/Spark-FP8",
        "revision": "a" * 40,
        "storage": "fp8_compressed_tensors",
        "output_dir": "checkpoints/sparkx2_5_fp8_source",
        "weights": {"model.safetensors": b"W" * 61},
        "assets": {
            "config.json": b'{"model_type":"spark2_5"}',
            "generation_config.json": b'{"do_sample":false}',
            "chat_template.jinja": b"{{ messages }}",
            "tokenizer.json": b'{"vocab":{}}',
            "configuration_spark.py": b"class Spark2_5Config: pass\n",
            "modeling_spark.py": b"class Spark2_5ForCausalLM: pass\n",
        },
        "weight_index": None,
    }
    spec.update(overrides)
    return spec


def make_fixture(root: Path, *, sources: list[dict] | None = None,
                 max_file_bytes: int | None = None,
                 max_asset_bytes: int | None = None,
                 max_snapshot_bytes: int | None = None,
                 total_budget: int | None = None,
                 min_free_bytes: int = 1024) -> dict:
    sources = sources or [default_source_spec()]
    pins_sources = []
    manifest_sources = []
    declared_sources = []
    payloads: dict[str, bytes] = {}
    weights_total = 0
    assets_total = 0
    snapshots: dict[str, int] = {}

    for spec in sources:
        weight_files = []
        for filename, data in spec["weights"].items():
            payloads[filename] = data
            weights_total += len(data)
            weight_files.append({
                "filename": filename,
                "bytes": len(data),
                "hub_lfs_sha256": _sha(data),
            })
        assets = []
        for filename, data in spec["assets"].items():
            payloads[filename] = data
            assets_total += len(data)
            assets.append({
                "filename": filename,
                "required": True,
                "retain": True,
                "expected_bytes": len(data),
                "expected_sha256": _sha(data),
            })
        index = spec.get("weight_index")
        if index is not None:
            if index not in spec["assets"]:
                raise AssertionError("fixture index must be an asset")
        snapshot_bytes = (sum(item["bytes"] for item in weight_files)
                          + sum(item["expected_bytes"] for item in assets))
        snapshots[spec["role"]] = snapshot_bytes
        pins_sources.append({
            "role": spec["role"],
            "repo_id": spec["repo_id"],
            "revision": spec["revision"],
            "storage": spec["storage"],
            "weight_files": weight_files,
            "weight_index": index,
            "quantization": ({"method": "compressed-tensors"}
                             if spec["storage"] == "fp8_compressed_tensors"
                             else None),
            "assets": assets,
        })
        manifest_sources.append({
            "role": spec["role"],
            "repo_id": spec["repo_id"],
            "revision": spec["revision"],
            "storage": spec["storage"],
            "weight_files": [dict(item, locally_verified=False,
                                  evidence="fixture header only")
                             for item in weight_files],
            "assets": [{"filename": item["filename"],
                        "bytes": item["expected_bytes"],
                        "sha256": item["expected_sha256"]}
                       for item in assets],
            "safetensors_headers": [{
                "filename": item["filename"],
                "header_bytes": 8,
                "header_sha256": _sha(b"header!!"),
                "payload_bytes": item["bytes"] - 16,
                "tensor_count": 1,
                "total_bytes": item["bytes"],
            } for item in weight_files],
            "tensor_inventory": {"parameter_elements_excluding_scales": 1234},
        })
        declared_sources.append({
            "role": spec["role"],
            "repo_id": spec["repo_id"],
            "revision": spec["revision"],
            "storage": spec["storage"],
            "output_dir": spec["output_dir"],
            "expected_snapshot_bytes": snapshot_bytes,
            "weight_file_count": len(weight_files),
            "asset_count": len(assets),
            "weight_index": index,
        })

    pins_path = root / "configs" / "source_plan_fixture.json"
    manifest_path = root / "reports" / "preparation_manifest_fixture.json"
    pins_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pins_path.write_text(json.dumps({
        "schema": "replica.spark_source_preparation_plan.v1",
        "weights_downloaded": False,
        "sources": pins_sources,
    }, indent=2))
    manifest_path.write_text(json.dumps({
        "schema": "replica.spark_source_preparation.v1",
        "status": "preparation_only_unqualified",
        "fetch_policy": {"weights_downloaded": False},
        "sources": manifest_sources,
    }, indent=2))

    planned_weights = weights_total
    planned_assets = assets_total
    planned_payload = planned_weights + planned_assets
    budget_total = (planned_payload + 3000) if total_budget is None else total_budget
    limits = {
        "max_file_bytes": (max(len(data) for spec in sources
                               for data in spec["weights"].values()) + 64
                           if max_file_bytes is None else max_file_bytes),
        "max_asset_bytes": (max(len(data) for spec in sources
                                for data in spec["assets"].values()) + 64
                            if max_asset_bytes is None else max_asset_bytes),
        "max_snapshot_bytes": (max(snapshots.values()) + 128
                               if max_snapshot_bytes is None else max_snapshot_bytes),
        "max_total_acquisition_bytes": budget_total,
        "min_free_bytes": min_free_bytes,
        "chunk_bytes": 7,
        "checkpoint_bytes": 14,
        "timeout_seconds": 5,
    }
    unallocated = budget_total - planned_payload - 1000 - 1000
    plan = {
        "schema": acq.SCHEMA,
        "reviewed_on": "2026-10-26",
        "scope": "synthetic test fixture",
        "authorization": {
            "user_authorized": True,
            "download_authorized": True,
            "model_execution": False,
            "remote_code_execution": False,
            "authorized_repo_ids": [spec["repo_id"] for spec in sources],
        },
        "pins": {
            "source_plan": {
                "path": "configs/source_plan_fixture.json",
                "schema": "replica.spark_source_preparation_plan.v1",
                "sha256": _sha(pins_path.read_bytes()),
            },
            "preparation_manifest": {
                "path": "reports/preparation_manifest_fixture.json",
                "schema": "replica.spark_source_preparation.v1",
                "sha256": _sha(manifest_path.read_bytes()),
            },
        },
        "access_order": [spec["role"] for spec in sources],
        "limits": limits,
        "budget": {
            "total_budget_bytes": budget_total,
            "scope": ["fixture"],
            "planned_weights_bytes": planned_weights,
            "planned_assets_bytes": planned_assets,
            "planned_payload_bytes": planned_payload,
            "environment_reserve_bytes": 1000,
            "evidence_reserve_bytes": 1000,
            "unallocated_bytes": unallocated,
        },
        "sources": declared_sources,
        "protocol": {},
        "environment": {},
        "data_access": {"split_order": ["train", "development", "heldout"],
                        "heldout_opened": False},
        "non_claims": ["fixture"],
    }
    plan_path = root / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2))
    return {
        "root": root,
        "plan_path": plan_path,
        "plan": plan,
        "pins_path": pins_path,
        "manifest_path": manifest_path,
        "payloads": payloads,
        "specs": sources,
        "snapshots": snapshots,
    }


def load_fixture(fixture: dict):
    return acq.load_acquisition_plan(fixture["plan_path"], root=fixture["root"])


def target_for(fixture: dict, spec: dict) -> Path:
    return fixture["root"] / spec["output_dir"]


def staging_for(fixture: dict, spec: dict) -> Path:
    return acq._staging_dir(target_for(fixture, spec), spec["revision"])


def seed_partial(staging: Path, *, filename: str, payload: bytes,
                 offset: int, extra_tail: bytes = b"",
                 corrupt_state: bool = False) -> None:
    staging.mkdir(parents=True, exist_ok=True)
    state = {
        "schema": "replica.spark_weight_acquisition_partial.v1",
        "filename": filename,
        "bytes": offset,
        "sha256": _sha(payload[:offset]),
        "expected_bytes": len(payload),
        "pinned_sha256": _sha(payload),
    }
    if corrupt_state:
        state["sha256"] = "0" * 64
    (staging / f"{filename}.part").write_bytes(payload[:offset] + extra_tail)
    (staging / f"{filename}.part.json").write_text(json.dumps(state))


def seed_staging_identity(plan, staging: Path, role: str) -> None:
    source = next(item for item in plan["sources"] if item["role"] == role)
    acq._write_staging_identity(staging, source)


def fingerprint_ok(manifest: dict) -> bool:
    body = {key: value for key, value in manifest.items()
            if key != "manifest_fingerprint"}
    return manifest.get("manifest_fingerprint") == acq.fingerprint(body)


# ---------------------------------------------------------------------------
# Pins and plan validation


def test_repository_plan_derives_the_official_pins():
    plan = acq.load_acquisition_plan(REAL_PLAN, root=ROOT)
    assert [source["role"] for source in plan["sources"]] == [
        "official_fp8_adaptation_base", "official_bf16_control"]
    fp8, bf16 = plan["sources"]
    assert fp8["revision"] == "0318d6f6ab070c7b5b43052db571489b2508006c"
    assert bf16["revision"] == "66e3ae7bee54e0a523d692cb8ef841f35409807b"
    assert fp8["snapshot_bytes"] == 4_466_408_296
    assert bf16["snapshot_bytes"] == 8_238_996_247
    assert fp8["weight_bytes"] == 4_451_626_120
    assert bf16["weight_bytes"] == 8_224_192_408
    fp8_weight = next(record for record in fp8["files"]
                      if record["weight_payload"])
    assert fp8_weight["sha256"] == (
        "f92114bf9af5588e0bcbc4fe1c4a0420fa17bba58199cb8b6cf1f3564e664be9")
    assert fp8["output_dir"] == "checkpoints/sparkx2_5_fp8_source"
    assert bf16["output_dir"] == "checkpoints/sparkx2_5_bf16_source"
    assert plan["budget"]["planned_payload_bytes"] == 12_705_404_543
    assert plan["budget"]["total_budget_bytes"] == 40 * 1024 ** 3
    for source in plan["sources"]:
        by_name = {record["filename"]: record for record in source["files"]}
        assert by_name["config.json"]["kind"] == "config"
        assert by_name["generation_config.json"]["kind"] == "generation_config"
        assert by_name["chat_template.jinja"]["kind"] == "chat_template"
        assert by_name["configuration_spark.py"]["kind"] == "remote_code"
        assert by_name["modeling_spark.py"]["kind"] == "remote_code"
        assert by_name["tokenizer.json"]["kind"] == "tokenizer"


def test_plan_rejects_tampered_pin_source_bytes(tmp_path):
    fixture = make_fixture(tmp_path)
    fixture["pins_path"].write_text(fixture["pins_path"].read_text() + " ")
    with pytest.raises(acq.PinMismatch):
        load_fixture(fixture)


def test_plan_rejects_inconsistent_weight_pin(tmp_path):
    fixture = make_fixture(tmp_path)
    pins = json.loads(fixture["pins_path"].read_text())
    pins["sources"][0]["weight_files"][0]["hub_lfs_sha256"] = "0" * 64
    fixture["pins_path"].write_text(json.dumps(pins))
    plan = fixture["plan"]
    plan["pins"]["source_plan"]["sha256"] = _sha(fixture["pins_path"].read_bytes())
    fixture["plan_path"].write_text(json.dumps(plan))
    with pytest.raises(acq.PinMismatch):
        load_fixture(fixture)


def test_plan_rejects_model_execution_authorization(tmp_path):
    fixture = make_fixture(tmp_path)
    plan = fixture["plan"]
    plan["authorization"]["model_execution"] = True
    fixture["plan_path"].write_text(json.dumps(plan))
    with pytest.raises(acq.PlanError):
        load_fixture(fixture)


def test_plan_rejects_per_file_cap_below_pin(tmp_path):
    fixture = make_fixture(tmp_path, max_file_bytes=10)
    with pytest.raises(acq.PlanError):
        load_fixture(fixture)


def test_plan_rejects_total_budget_below_pins(tmp_path):
    fixture = make_fixture(tmp_path)
    planned = fixture["plan"]["budget"]["planned_payload_bytes"]
    fixture = make_fixture(tmp_path, total_budget=planned - 1)
    with pytest.raises(acq.PlanError):
        load_fixture(fixture)


def test_plan_rejects_escaping_output_dir(tmp_path):
    fixture = make_fixture(tmp_path, sources=[
        default_source_spec(output_dir="../escape")])
    with pytest.raises(acq.PlanError):
        load_fixture(fixture)


def test_script_never_imports_or_executes_model_code():
    text = SCRIPT.read_text()
    for forbidden in ("import torch", "from torch", "import transformers",
                      "from transformers", "import safetensors",
                      "from safetensors", "from_pretrained", "exec(",
                      "subprocess"):
        assert forbidden not in text


# ---------------------------------------------------------------------------
# Successful acquisition


def test_successful_acquisition_is_atomic_and_records_manifest(tmp_path):
    fixture = make_fixture(tmp_path)
    plan = load_fixture(fixture)
    free = FreeBytes(10 ** 9)
    transport = FakeTransport(fixture["payloads"])
    summary = acq.run_acquisition(plan, transport=transport, free_bytes_fn=free,
                                  receipt=tmp_path / "receipt.json")
    assert summary["status"] == "complete"
    assert summary["model_executed"] is False
    assert summary["remote_code_executed"] is False
    spec = fixture["specs"][0]
    target = target_for(fixture, spec)
    staging = staging_for(fixture, spec)
    assert target.is_dir()
    assert not staging.exists()
    result = summary["sources"][0]
    assert result["staging_removed"] is True
    assert result["bytes_downloaded"] == fixture["snapshots"][spec["role"]]
    assert result["free_bytes_before"] == 10 ** 9
    assert result["high_water_estimate_bytes"] >= result["final_bytes"]

    names = sorted(path.name for path in target.iterdir())
    expected = sorted(list(spec["weights"]) + list(spec["assets"])
                      + ["acquisition_manifest.json"])
    assert names == expected
    for filename, data in fixture["payloads"].items():
        assert (target / filename).read_bytes() == data

    manifest = json.loads((target / "acquisition_manifest.json").read_text())
    assert fingerprint_ok(manifest)
    assert manifest["schema"] == acq.MANIFEST_SCHEMA
    assert manifest["status"] == "complete"
    assert manifest["weight_payload_present"] is True
    assert manifest["header_only"] is False
    assert manifest["snapshot"]["weight_payload_present"] is True
    assert manifest["snapshot"]["header_only"] is False
    assert manifest["snapshot"]["weight_bytes"] == len(
        spec["weights"]["model.safetensors"])
    assert manifest["snapshot"]["parameter_elements_excluding_scales"] == 1234
    assert manifest["header_evidence"]["header_only"] is True
    assert manifest["header_evidence"]["payload_downloaded_during_preparation"] is False
    shard = manifest["header_evidence"]["shards"][0]
    assert shard["payload_bytes"] == len(spec["weights"]["model.safetensors"]) - 16
    assert shard["header_prefix_bytes"] == 8

    by_name = {record["filename"]: record for record in manifest["files"]}
    weight = by_name["model.safetensors"]
    assert weight["kind"] == "weights"
    assert weight["payload_class"] == "weight_shard"
    assert weight["weight_payload"] is True
    assert weight["sha256"] == _sha(fixture["payloads"]["model.safetensors"])
    assert weight["size_verified"] is True and weight["sha256_verified"] is True
    config = by_name["config.json"]
    assert config["kind"] == "config" and config["weight_payload"] is False
    assert by_name["configuration_spark.py"]["kind"] == "remote_code"
    assert manifest["remote_code"]["executed"] is False
    assert manifest["remote_code"]["files"][0]["executed"] is False
    assert manifest["verification"]["all_files_verified"] is True
    assert manifest["verification"]["final_hash_verification"] == "exact_sha256_full_file"
    assert manifest["acquisition"]["atomic_staging"] is True
    assert manifest["acquisition"]["staging_removed"] is True
    assert manifest["acquisition"]["duplicate_completed_payload"] is False
    assert manifest["acquisition"]["free_bytes_before"] == 10 ** 9
    assert manifest["acquisition"]["free_bytes_after"] == 10 ** 9
    assert manifest["acquisition"]["high_water_estimate_bytes"] >= \
        manifest["snapshot"]["total_bytes"]

    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["status"] == "complete"
    assert receipt["budget"]["bytes_downloaded"] == fixture["snapshots"][spec["role"]]
    assert receipt["budget"]["final_bytes"] == fixture["snapshots"][spec["role"]]
    assert receipt["free_bytes_before"] == 10 ** 9
    assert receipt["free_bytes_after"] == 10 ** 9


def test_weight_index_is_manifested_but_not_weight_payload(tmp_path):
    source = default_source_spec(
        weight_index="model.safetensors.index.json",
        assets={
            "config.json": b'{"model_type":"spark2_5"}',
            "generation_config.json": b"{}",
            "chat_template.jinja": b"chat",
            "tokenizer.json": b'{"v":1}',
            "configuration_spark.py": b"pass\n",
            "modeling_spark.py": b"pass\n",
            "model.safetensors.index.json": b'{"weight_map":{}}',
        })
    fixture = make_fixture(tmp_path, sources=[source])
    plan = load_fixture(fixture)
    summary = acq.run_acquisition(plan, transport=FakeTransport(fixture["payloads"]),
                                  free_bytes_fn=FreeBytes())
    assert summary["status"] == "complete"
    target = target_for(fixture, source)
    manifest = json.loads((target / "acquisition_manifest.json").read_text())
    record = next(item for item in manifest["files"]
                  if item["filename"] == "model.safetensors.index.json")
    assert record["kind"] == "weights_index"
    assert record["payload_class"] == "weight_index"
    assert record["weight_payload"] is False
    assert manifest["snapshot"]["weight_payload_files"] == 1
    assert manifest["snapshot"]["weight_bytes"] == len(source["weights"]["model.safetensors"])


# ---------------------------------------------------------------------------
# Caps and failures


def test_oversized_declared_body_is_refused_before_reading(tmp_path):
    fixture = make_fixture(tmp_path)
    plan = load_fixture(fixture)
    weight = "model.safetensors"
    transport = FakeTransport(
        fixture["payloads"],
        declared_lengths={weight: len(fixture["payloads"][weight]) + 10 ** 6})
    summary = acq.run_acquisition(plan, transport=transport,
                                  free_bytes_fn=FreeBytes())
    assert summary["status"] == "failed"
    assert summary["failures"][0]["error_type"] == "FetchRefused"
    assert "content_length_mismatch" in summary["failures"][0]["evidence"]["reason"]
    target = target_for(fixture, fixture["specs"][0])
    assert not target.exists()
    receipts = list(target.parent.glob("*.failure-*.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text())
    assert receipt["status"] == "failed"
    assert receipt["staging_preserved"] is True
    assert receipt["target_created"] is False
    assert receipt["auto_deleted_user_evidence"] is False


def test_hash_mismatch_is_quarantined_and_never_finalized(tmp_path):
    fixture = make_fixture(tmp_path)
    plan = load_fixture(fixture)
    bad = bytearray(fixture["payloads"]["model.safetensors"])
    bad[0] ^= 0xFF
    transport = FakeTransport({**fixture["payloads"],
                               "model.safetensors": bytes(bad)})
    summary = acq.run_acquisition(plan, transport=transport,
                                  free_bytes_fn=FreeBytes())
    assert summary["status"] == "failed"
    failure = summary["failures"][0]
    assert failure["error_type"] == "HashMismatch"
    assert "quarantined" in failure["evidence"]["detail"]
    target = target_for(fixture, fixture["specs"][0])
    assert not target.exists()
    staging = staging_for(fixture, fixture["specs"][0])
    quarantined = list(staging.glob("model.safetensors.part.corrupt-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == bytes(bad)
    receipt = json.loads(Path(failure["failure_receipt"]).read_text())
    assert receipt["staging_preserved"] is True
    assert receipt["partials"] == []


def test_insufficient_free_space_fails_before_payload(tmp_path):
    fixture = make_fixture(tmp_path, min_free_bytes=1024)
    plan = load_fixture(fixture)
    transport = FakeTransport(fixture["payloads"])
    summary = acq.run_acquisition(plan, transport=transport,
                                  free_bytes_fn=FreeBytes(10))
    assert summary["status"] == "failed"
    assert summary["failures"][0]["error_type"] == "BudgetExceeded"
    assert transport.requests == 0
    assert not target_for(fixture, fixture["specs"][0]).exists()


def test_failures_preserve_distinct_receipts(tmp_path):
    fixture = make_fixture(tmp_path)
    plan = load_fixture(fixture)
    result = None
    for _ in range(2):
        result = acq.run_acquisition(
            plan, transport=FakeTransport(fixture["payloads"]),
            free_bytes_fn=FreeBytes(1))
    target = target_for(fixture, fixture["specs"][0])
    receipts = list(target.parent.glob("*.failure-*.json"))
    assert len(receipts) == 2
    assert all(json.loads(path.read_text())["status"] == "failed"
               for path in receipts)
    assert result["status"] == "failed"


def test_existing_target_without_manifest_is_not_overwritten(tmp_path):
    fixture = make_fixture(tmp_path)
    plan = load_fixture(fixture)
    target = target_for(fixture, fixture["specs"][0])
    target.mkdir(parents=True)
    (target / "keep.txt").write_text("user evidence")
    summary = acq.run_acquisition(plan, transport=ExplodingTransport(),
                                  free_bytes_fn=FreeBytes())
    assert summary["status"] == "failed"
    assert summary["failures"][0]["error_type"] == "SnapshotConflict"
    assert (target / "keep.txt").read_text() == "user evidence"


# ---------------------------------------------------------------------------
# Resume


def test_resume_rejects_corrupt_partial_without_network(tmp_path):
    fixture = make_fixture(tmp_path)
    plan = load_fixture(fixture)
    spec = fixture["specs"][0]
    staging = staging_for(fixture, spec)
    seed_staging_identity(plan, staging, spec["role"])
    payload = fixture["payloads"]["model.safetensors"]
    seed_partial(staging, filename="model.safetensors", payload=payload,
                 offset=10, corrupt_state=True)
    transport = FakeTransport(fixture["payloads"])
    summary = acq.run_acquisition(plan, transport=transport,
                                  free_bytes_fn=FreeBytes())
    assert summary["status"] == "failed"
    failure = summary["failures"][0]
    assert failure["error_type"] == "PartialCorrupt"
    assert "SHA256" in failure["evidence"]["detail"] or \
        "sha256" in json.dumps(failure["evidence"]).lower()
    assert all(name != "model.safetensors" for name, _ in transport.calls)
    assert not target_for(fixture, spec).exists()
    assert (staging / "model.safetensors.part").exists()
    assert (staging / "model.safetensors.part.json").exists()


def test_partial_bytes_without_state_are_refused(tmp_path):
    fixture = make_fixture(tmp_path)
    plan = load_fixture(fixture)
    spec = fixture["specs"][0]
    staging = staging_for(fixture, spec)
    seed_staging_identity(plan, staging, spec["role"])
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "model.safetensors.part").write_bytes(b"unverified")
    transport = FakeTransport(fixture["payloads"])
    summary = acq.run_acquisition(plan, transport=transport,
                                  free_bytes_fn=FreeBytes())
    assert summary["status"] == "failed"
    assert summary["failures"][0]["error_type"] == "PartialCorrupt"
    assert all(name != "model.safetensors" for name, _ in transport.calls)


def test_resume_from_verified_prefix_uses_range_and_discards_tail(tmp_path):
    fixture = make_fixture(tmp_path)
    plan = load_fixture(fixture)
    spec = fixture["specs"][0]
    staging = staging_for(fixture, spec)
    seed_staging_identity(plan, staging, spec["role"])
    payload = fixture["payloads"]["model.safetensors"]
    seed_partial(staging, filename="model.safetensors", payload=payload,
                 offset=10, extra_tail=b"TAIL")
    transport = FakeTransport(fixture["payloads"])
    summary = acq.run_acquisition(plan, transport=transport,
                                  free_bytes_fn=FreeBytes())
    assert summary["status"] == "complete"
    result = summary["sources"][0]
    assert result["resumed_from_partial"] is True
    assert result["resumed_bytes"] == 10
    assert ("model.safetensors", 10) in transport.calls
    target = target_for(fixture, spec)
    assert (target / "model.safetensors").read_bytes() == payload
    manifest = json.loads((target / "acquisition_manifest.json").read_text())
    assert manifest["acquisition"]["resumed_from_partial"] is True
    assert manifest["acquisition"]["resumed_bytes"] == 10
    assert manifest["acquisition"]["discarded_unverified_tail_bytes"] == 4


# ---------------------------------------------------------------------------
# Idempotence and offline verification


def test_completed_snapshot_is_verified_in_place_without_network(tmp_path):
    fixture = make_fixture(tmp_path)
    plan = load_fixture(fixture)
    first = acq.run_acquisition(plan, transport=FakeTransport(fixture["payloads"]),
                                free_bytes_fn=FreeBytes())
    assert first["status"] == "complete"
    target = target_for(fixture, fixture["specs"][0])
    manifest_path = target / "acquisition_manifest.json"
    before = manifest_path.read_bytes()
    files_before = sorted(path.name for path in target.iterdir())

    second_transport = ExplodingTransport()
    second = acq.run_acquisition(plan, transport=second_transport,
                                 free_bytes_fn=FreeBytes())
    assert second["status"] == "complete"
    result = second["sources"][0]
    assert result["status"] == "already_complete"
    assert result["reused_completed_snapshot"] is True
    assert result["network_requests"] == 0
    assert result["bytes_downloaded"] == 0
    assert second_transport.requests == 0
    assert manifest_path.read_bytes() == before
    assert sorted(path.name for path in target.iterdir()) == files_before
    assert sum(1 for path in target.iterdir()
               if path.name == "model.safetensors") == 1


def test_corrupt_completed_snapshot_is_reported_not_repaired(tmp_path):
    fixture = make_fixture(tmp_path)
    plan = load_fixture(fixture)
    acq.run_acquisition(plan, transport=FakeTransport(fixture["payloads"]),
                        free_bytes_fn=FreeBytes())
    target = target_for(fixture, fixture["specs"][0])
    with (target / "model.safetensors").open("ab") as stream:
        stream.write(b"x")
    summary = acq.run_acquisition(plan, transport=ExplodingTransport(),
                                  free_bytes_fn=FreeBytes())
    assert summary["status"] == "failed"
    assert summary["failures"][0]["error_type"] == "SnapshotConflict"
    assert target.is_dir()
    assert (target / "acquisition_manifest.json").is_file()


def test_offline_cli_validates_without_network(tmp_path):
    fixture = make_fixture(tmp_path)
    assert acq.main(["--plan", str(fixture["plan_path"]),
                     "--root", str(fixture["root"]),
                     "--offline", "--quiet"]) == 2
    plan = load_fixture(fixture)
    acq.run_acquisition(plan, transport=FakeTransport(fixture["payloads"]),
                        free_bytes_fn=FreeBytes())
    assert acq.main(["--plan", str(fixture["plan_path"]),
                     "--root", str(fixture["root"]),
                     "--offline", "--quiet"]) == 0


def test_dry_run_creates_nothing(tmp_path):
    fixture = make_fixture(tmp_path)
    assert acq.main(["--plan", str(fixture["plan_path"]),
                     "--root", str(fixture["root"]),
                     "--dry-run", "--quiet"]) == 0
    assert not (tmp_path / "checkpoints").exists()


def test_two_snapshots_report_high_water_and_budget(tmp_path):
    shared_assets = {
        "config.json": b'{"model_type":"spark2_5"}',
        "generation_config.json": b'{"do_sample":false}',
        "chat_template.jinja": b"{{ messages }}",
        "tokenizer.json": b'{"vocab":{}}',
        "configuration_spark.py": b"class Spark2_5Config: pass\n",
        "modeling_spark.py": b"class Spark2_5ForCausalLM: pass\n",
    }
    fp8 = default_source_spec()
    bf16 = default_source_spec(
        role="official_bf16_control",
        repo_id="Example/Spark-BF16",
        revision="b" * 40,
        storage="bf16",
        output_dir="checkpoints/sparkx2_5_bf16_source",
        weights={"model-00001-of-00001.safetensors": b"B" * 80},
        weight_index="model.safetensors.index.json",
        assets={**shared_assets,
                "model.safetensors.index.json": b'{"weight_map":{}}'})
    fixture = make_fixture(tmp_path, sources=[fp8, bf16])
    plan = load_fixture(fixture)
    transport = FakeTransport(fixture["payloads"])
    summary = acq.run_acquisition(plan, transport=transport,
                                  free_bytes_fn=FreeBytes())
    assert summary["status"] == "complete"
    assert [item["role"] for item in summary["sources"]] == [
        fp8["role"], bf16["role"]]
    first_bytes = fixture["snapshots"][fp8["role"]]
    second_bytes = fixture["snapshots"][bf16["role"]]
    assert summary["budget"]["final_bytes"] == first_bytes + second_bytes
    assert summary["budget"]["high_water_estimate_bytes"] == first_bytes + second_bytes
    assert summary["budget"]["bytes_downloaded"] == first_bytes + second_bytes
    bf16_manifest = json.loads(
        (target_for(fixture, bf16) / "acquisition_manifest.json").read_text())
    assert bf16_manifest["acquisition"]["prior_completed_bytes"] == first_bytes
    assert bf16_manifest["acquisition"]["high_water_estimate_bytes"] == \
        first_bytes + second_bytes
    assert fingerprint_ok(bf16_manifest)
    for spec in (fp8, bf16):
        assert target_for(fixture, spec).is_dir()
        assert not staging_for(fixture, spec).exists()


def test_existing_receipt_path_is_refused_before_network(tmp_path):
    fixture = make_fixture(tmp_path)
    plan = load_fixture(fixture)
    receipt = tmp_path / "receipt.json"
    receipt.write_text("previous evidence")
    transport = ExplodingTransport()
    with pytest.raises(acq.SnapshotConflict):
        acq.run_acquisition(plan, transport=transport,
                            free_bytes_fn=FreeBytes(), receipt=receipt)
    assert transport.requests == 0
    assert receipt.read_text() == "previous evidence"

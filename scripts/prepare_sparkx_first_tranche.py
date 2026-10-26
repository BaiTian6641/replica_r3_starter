#!/usr/bin/env python3
"""Bounded first-tranche admission for already-local or owned synthetic assets.

This is a protocol implementation, not an acquisition tool. It never touches
the network, a model, a dataset loader or a native runtime. It reads a pinned
configuration, a JSONL asset manifest and the exact asset bytes, then applies:

* field, revision, license, gate and provenance checks,
* exact SHA-256 and size verification,
* exact-duplicate deduplication,
* group-key leakage prevention and deterministic split assignment,
* per-asset, per-source and tranche caps,
* explicit source/tranche stop rules.

Real external sources are pinned but not gate-cleared in this task, so any
real-source asset is refused. The committed fixture set is repository-owned
synthetic evidence and is labelled as such; it is not external media.

Usage:
  python3 scripts/prepare_sparkx_first_tranche.py
  python3 scripts/prepare_sparkx_first_tranche.py --check
  python3 scripts/prepare_sparkx_first_tranche.py --root . --dry-run

Exit codes: 0 tranche admitted (at least the configured minimum),
1 tranche refused by a stop rule, 2 invalid configuration or missing input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

FORMAT = "replica.sparkx_first_tranche_receipt.v1"
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = "configs/sparkx_first_tranche_v1.json"
SPLITS = ("train", "development", "audit")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class TrancheError(RuntimeError):
    """Configuration or input is structurally unusable."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def is_pinned_revision(value: object) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{40}", value))


def assign_split(group_key: str, salt: str, caps: dict) -> str:
    """Deterministic 70/15/15 assignment from the group key alone."""
    digest = hashlib.sha256(f"{salt}\x00{group_key}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10000
    train = int(caps["train_basis_points"])
    development = train + int(caps["development_basis_points"])
    if bucket < train:
        return "train"
    if bucket < development:
        return "development"
    return "audit"


def validate_config(config: dict) -> dict:
    required = (
        "caps",
        "split_assignment",
        "dedup",
        "leakage",
        "source_pins",
        "synthetic_fixtures",
        "asset_requirements",
        "stop_rules",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise TrancheError(f"configuration missing fields: {missing}")
    if config.get("format") != "replica.sparkx_first_tranche.v1":
        raise TrancheError(f"unsupported configuration format: {config.get('format')!r}")
    if config.get("bulk_download") is not False or config.get("download_authorized") is not False:
        raise TrancheError("first tranche must not authorize bulk download")

    caps = config["caps"]
    for key in (
        "max_assets",
        "max_assets_per_source",
        "max_asset_bytes",
        "max_total_bytes",
        "min_admitted_assets",
    ):
        if not isinstance(caps.get(key), int) or caps[key] <= 0:
            raise TrancheError(f"invalid cap: {key}")
    if caps["train_basis_points"] + caps["development_basis_points"] + caps["audit_basis_points"] != 10000:
        raise TrancheError("split basis points must sum to 10000")
    if not 0 <= float(caps["max_duplicate_rate_per_source"]) <= 1:
        raise TrancheError("max_duplicate_rate_per_source must be in [0, 1]")

    pins = config["source_pins"]
    if not isinstance(pins, list) or not pins:
        raise TrancheError("source_pins must be a non-empty list")
    seen: set[str] = set()
    for pin in pins:
        source_id = pin.get("id")
        if not isinstance(source_id, str) or not source_id:
            raise TrancheError("source pin missing id")
        if source_id in seen:
            raise TrancheError(f"duplicate source pin: {source_id}")
        seen.add(source_id)
        if not is_pinned_revision(pin.get("revision")):
            raise TrancheError(f"source {source_id} revision is not a 40-hex pin")
        if not pin.get("repo_id") or not pin.get("license_metadata"):
            raise TrancheError(f"source {source_id} missing repo_id or license metadata")
        if pin.get("gate") != "cleared" and pin.get("gate") != "not_acquired_or_cleared":
            raise TrancheError(f"source {source_id} has unknown gate state")
        if pin.get("priority") == "EVAL" or pin.get("role") == "evaluation_only":
            raise TrancheError(f"evaluation-only source must not be admitted: {source_id}")

    synthetic = config["synthetic_fixtures"]
    if synthetic.get("label") != "owned_synthetic_not_external_media":
        raise TrancheError("synthetic fixture label must state that it is not external media")
    if not synthetic.get("revision") or not synthetic.get("source_id"):
        raise TrancheError("synthetic fixture source_id/revision missing")

    requirements = config["asset_requirements"]
    required_fields = requirements.get("required_fields")
    if not isinstance(required_fields, list) or not required_fields:
        raise TrancheError("asset_requirements.required_fields must be non-empty")

    return config


def load_assets(path: Path) -> list[dict]:
    if not path.is_file():
        raise TrancheError(f"asset manifest missing: {path}")
    assets = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as error:
            raise TrancheError(f"asset manifest line {line_number}: {error}") from error
        if not isinstance(record, dict):
            raise TrancheError(f"asset manifest line {line_number}: not an object")
        assets.append(record)
    if not assets:
        raise TrancheError("asset manifest is empty")
    return assets


def _license_allowed(license_value: object, policy: dict) -> bool:
    if not isinstance(license_value, str) or not license_value:
        return False
    if license_value in policy.get("allow", []):
        return True
    return False


@dataclass
class Admission:
    config: dict
    root: Path
    asset_root: Path
    synthetic_root: Path
    source_by_id: dict
    admitted: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    source_stats: dict = field(default_factory=dict)
    source_stopped: set = field(default_factory=set)
    stop_reasons: list = field(default_factory=list)
    global_leakage: bool = False
    cap_stopped: bool = False
    seen_asset_ids: set = field(default_factory=set)
    seen_paths: set = field(default_factory=set)
    sha_owner: dict = field(default_factory=dict)
    group_split: dict = field(default_factory=dict)
    group_source: dict = field(default_factory=dict)
    admitted_bytes: int = 0

    def reject(self, asset: dict, reason: str, detail: str = "") -> None:
        self.rejected.append(
            {
                "asset_id": asset.get("asset_id"),
                "source_id": asset.get("source_id"),
                "path": asset.get("path"),
                "reason": reason,
                "detail": detail,
            }
        )

    def stop(self, code: str, scope: str, detail: str) -> None:
        entry = {"code": code, "scope": scope, "detail": detail}
        if entry not in self.stop_reasons:
            self.stop_reasons.append(entry)

    def _source_stats(self, source_id: str) -> dict:
        return self.source_stats.setdefault(
            source_id, {"attempted": 0, "admitted": 0, "duplicates": 0}
        )

    def admit_one(self, asset: dict) -> None:
        source_id = asset.get("source_id")
        stats = self._source_stats(source_id if isinstance(source_id, str) else "<missing>")
        stats["attempted"] += 1

        if source_id in self.source_stopped:
            self.reject(asset, "source_stopped", "an earlier integrity failure stopped this source")
            return

        requirements = self.config["asset_requirements"]
        missing = [
            field
            for field in requirements["required_fields"]
            if field not in asset or asset[field] in (None, "", [], {})
        ]
        if missing:
            self.reject(asset, "missing_required_fields", ",".join(missing))
            self._integrity_failure(source_id, "missing required fields")
            return

        asset_id = asset["asset_id"]
        if not isinstance(asset_id, str) or not asset_id:
            self.reject(asset, "invalid_asset_id")
            self._integrity_failure(source_id, "invalid asset id")
            return
        if not isinstance(source_id, str) or not source_id:
            self.reject(asset, "invalid_source_id")
            self._integrity_failure("<missing>", "invalid source id")
            return
        if not isinstance(asset["group_key"], str) or not asset["group_key"]:
            self.reject(asset, "invalid_group_key")
            self._integrity_failure(source_id, "invalid group key")
            return
        if asset_id in self.seen_asset_ids:
            self.reject(asset, "duplicate_asset_id")
            self._count_duplicate(source_id)
            return
        self.seen_asset_ids.add(asset_id)

        if not isinstance(asset["bytes"], int) or asset["bytes"] <= 0:
            self.reject(asset, "invalid_byte_count")
            self._integrity_failure(source_id, "invalid byte count")
            return
        if not isinstance(asset["sha256"], str) or not SHA256_PATTERN.fullmatch(asset["sha256"]):
            self.reject(asset, "invalid_sha256")
            self._integrity_failure(source_id, "invalid sha256")
            return

        if asset["media_kind"] not in requirements["allowed_media_kinds"]:
            self.reject(asset, "disallowed_media_kind")
            self._integrity_failure(source_id, "disallowed media kind")
            return

        if asset["synthetic"] is True:
            synthetic = self.config["synthetic_fixtures"]
            if source_id != synthetic["source_id"]:
                self.reject(asset, "unknown_synthetic_source")
                self._integrity_failure(source_id, "unknown synthetic source")
                return
            if asset["source_revision"] != synthetic["revision"]:
                self.reject(asset, "synthetic_revision_mismatch")
                self._integrity_failure(source_id, "synthetic revision mismatch")
                return
            if asset["license"] != synthetic["license_claim"]:
                self.reject(asset, "synthetic_license_mismatch")
                self._integrity_failure(source_id, "synthetic license mismatch")
                return
        else:
            pin = self.source_by_id.get(source_id)
            if pin is None:
                self.reject(asset, "unknown_source")
                self._integrity_failure(source_id, "unknown source")
                return
            if asset["source_revision"] != pin["revision"]:
                self.reject(asset, "source_revision_mismatch")
                self._integrity_failure(source_id, "source revision mismatch")
                return
            if pin.get("gate") != "cleared":
                self.reject(asset, "source_not_acquired_or_cleared", pin.get("gate", ""))
                self.stop(
                    "external_media_not_acquired",
                    "tranche",
                    "no pinned external source has a clearing decision; only synthetic fixtures may pass",
                )
                self._integrity_failure(source_id, "source not cleared")
                return
            if not _license_allowed(asset["license"], self.config["license_policy"]):
                self.reject(asset, "disallowed_license", str(asset["license"]))
                self._integrity_failure(source_id, "disallowed license")
                return

        if asset["synthetic"] is True:
            synthetic = self.config["synthetic_fixtures"]
            expected_root = (self.root / synthetic["asset_root"]).resolve()
        else:
            expected_root = self.asset_root
        raw_path = asset["path"]
        if not isinstance(raw_path, str) or Path(raw_path).is_absolute() or ".." in Path(raw_path).parts:
            self.reject(asset, "path_outside_asset_root", "path must be repository-relative")
            self._integrity_failure(source_id, "path traversal")
            return
        candidate = (self.root / raw_path).resolve()
        try:
            candidate.relative_to(expected_root)
        except ValueError:
            self.reject(asset, "path_outside_asset_root", raw_path)
            self._integrity_failure(source_id, "path outside root")
            return
        if raw_path in self.seen_paths:
            self.reject(asset, "duplicate_path")
            self._count_duplicate(source_id)
            return
        if not candidate.is_file():
            self.reject(asset, "missing_asset_file", raw_path)
            self._integrity_failure(source_id, "missing asset file")
            return
        self.seen_paths.add(raw_path)

        if asset["bytes"] > self.config["caps"]["max_asset_bytes"]:
            self.reject(asset, "asset_too_large", str(asset["bytes"]))
            self.stop(
                "global_cap_exceeded",
                "tranche",
                f"asset {asset_id} exceeds max_asset_bytes",
            )
            self.cap_stopped = True
            return

        actual_bytes = candidate.stat().st_size
        actual_sha = sha256_file(candidate)
        if actual_bytes != asset["bytes"] or actual_sha != asset["sha256"]:
            self.reject(
                asset,
                "hash_or_size_mismatch",
                f"expected {asset['sha256']}/{asset['bytes']}, got {actual_sha}/{actual_bytes}",
            )
            self._integrity_failure(source_id, "hash or size mismatch")
            return

        if asset["sha256"] in self.sha_owner:
            self.reject(
                asset,
                "duplicate_exact_media",
                f"first admitted by {self.sha_owner[asset['sha256']]}",
            )
            self._count_duplicate(source_id)
            return

        group_key = asset["group_key"]
        if group_key in self.group_source and self.group_source[group_key] != source_id:
            self.reject(asset, "group_source_collision", group_key)
            self._leakage(source_id, f"group {group_key} appears under multiple sources")
            return
        self.group_source.setdefault(group_key, source_id)

        split = assign_split(
            group_key, self.config["split_assignment"]["salt"], self.config["caps"]
        )
        if group_key in self.group_split and self.group_split[group_key] != split:
            self.reject(asset, "split_leakage", group_key)
            self._leakage(source_id, f"group {group_key} maps to multiple splits")
            return
        declared = asset.get("declared_split")
        if declared is not None and declared != split:
            self.reject(asset, "declared_split_conflict", f"{declared} != {split}")
            self._leakage(source_id, f"declared split conflicts for {group_key}")
            return
        self.group_split.setdefault(group_key, split)

        if stats["admitted"] >= self.config["caps"]["max_assets_per_source"]:
            self.reject(asset, "source_cap_exceeded")
            self.stop("global_cap_exceeded", "tranche", f"source {source_id} cap")
            self.cap_stopped = True
            return
        if len(self.admitted) >= self.config["caps"]["max_assets"]:
            self.reject(asset, "tranche_cap_exceeded")
            self.stop("global_cap_exceeded", "tranche", "max_assets")
            self.cap_stopped = True
            return
        if self.admitted_bytes + actual_bytes > self.config["caps"]["max_total_bytes"]:
            self.reject(asset, "tranche_byte_cap_exceeded")
            self.stop("global_cap_exceeded", "tranche", "max_total_bytes")
            self.cap_stopped = True
            return

        admitted = {
            "asset_id": asset_id,
            "source_id": source_id,
            "source_revision": asset["source_revision"],
            "group_key": group_key,
            "sha256": actual_sha,
            "bytes": actual_bytes,
            "media_kind": asset["media_kind"],
            "license": asset["license"],
            "path": raw_path,
            "provenance": asset["provenance"],
            "synthetic": asset["synthetic"],
            "split": split,
        }
        self.admitted.append(admitted)
        self.sha_owner[asset["sha256"]] = asset_id
        self.admitted_bytes += actual_bytes
        stats["admitted"] += 1

    def _count_duplicate(self, source_id: object) -> None:
        stats = self._source_stats(str(source_id))
        stats["duplicates"] += 1
        attempted = stats["attempted"]
        rate = stats["duplicates"] / attempted if attempted else 0.0
        if rate > float(self.config["caps"]["max_duplicate_rate_per_source"]):
            self.source_stopped.add(source_id)
            self.stop(
                "duplicate_rate_exceeded",
                "source",
                f"{source_id}: {stats['duplicates']}/{attempted} exact duplicates",
            )

    def _integrity_failure(self, source_id: object, detail: str) -> None:
        if isinstance(source_id, str):
            self.source_stopped.add(source_id)
            self.stop("source_integrity_failure", "source", f"{source_id}: {detail}")

    def _leakage(self, source_id: object, detail: str) -> None:
        self.global_leakage = True
        self.stop("global_leakage", "tranche", detail)
        self._integrity_failure(source_id, detail)

    def finish(self) -> tuple[list[dict], list[dict], dict]:
        if self.global_leakage:
            for admitted in self.admitted:
                self.rejected.append(
                    {
                        "asset_id": admitted["asset_id"],
                        "source_id": admitted["source_id"],
                        "path": admitted["path"],
                        "reason": "tranche_refused_global_leakage",
                        "detail": "global leakage stop rule",
                    }
                )
            self.admitted = []
            self.admitted_bytes = 0

        if len(self.admitted) < self.config["caps"]["min_admitted_assets"]:
            self.stop(
                "below_minimum",
                "tranche",
                f"{len(self.admitted)} < {self.config['caps']['min_admitted_assets']}",
            )

        splits = {
            split: {"assets": 0, "bytes": 0}
            for split in SPLITS
        }
        for admitted in self.admitted:
            splits[admitted["split"]]["assets"] += 1
            splits[admitted["split"]]["bytes"] += admitted["bytes"]

        admitted = sorted(self.admitted, key=lambda item: item["asset_id"])
        rejected = sorted(
            self.rejected, key=lambda item: (item.get("asset_id") or "", item["reason"])
        )
        receipt = {
            "format": FORMAT,
            "kind": "bounded_tranche_protocol_receipt_not_training_admission",
            "deterministic": True,
            "synthetic_only": all(item["synthetic"] for item in admitted),
            "external_media_acquired": False,
            "training_admitted": False,
            "config": {
                "path": self.config.get("_path", CONFIG_PATH),
                "sha256": self.config.get("_sha256"),
            },
            "asset_manifest": {
                "path": self.config.get("_asset_manifest", ""),
                "sha256": self.config.get("_asset_manifest_sha256"),
            },
            "admitted_count": len(admitted),
            "rejected_count": len(rejected),
            "admitted_bytes": sum(item["bytes"] for item in admitted),
            "splits": splits,
            "dedup": {
                "exact_sha256": True,
                "duplicates_rejected": sum(1 for item in rejected if item["reason"] == "duplicate_exact_media"),
            },
            "source_stats": dict(sorted(self.source_stats.items())),
            "source_stopped": sorted(str(value) for value in self.source_stopped),
            "stopped": bool(self.stop_reasons),
            "stop_reasons": sorted(self.stop_reasons, key=lambda item: (item["code"], item["scope"], item["detail"])),
            "status": (
                "admitted_synthetic_protocol_fixtures"
                if len(admitted) >= self.config["caps"]["min_admitted_assets"]
                else "refused_no_admission"
            ),
            "non_claims": list(self.config.get("non_claims", [])),
            "network_used": False,
            "native_execution": False,
        }
        if not admitted:
            receipt["status"] = "refused_no_admission"
        return admitted, rejected, receipt


def run(
    root: Path | None = None,
    config_path: str | Path = CONFIG_PATH,
    asset_manifest_path: str | Path | None = None,
) -> tuple[list[dict], list[dict], dict]:
    root = Path(root) if root is not None else ROOT
    root_resolved = root.resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    if not config_file.is_file():
        raise TrancheError(f"configuration missing: {config_file}")
    try:
        config_rel = str(config_file.resolve().relative_to(root_resolved))
    except ValueError:
        config_rel = config_file.name
    config_sha = sha256_file(config_file)
    config = validate_config(json.loads(config_file.read_text(encoding="utf-8")))
    config["_path"] = config_rel
    config["_sha256"] = config_sha

    synthetic = config["synthetic_fixtures"]
    asset_root = (root / synthetic["asset_root"]).resolve()
    manifest_file = Path(
        asset_manifest_path
        if asset_manifest_path is not None
        else synthetic["asset_manifest"]
    )
    if not manifest_file.is_absolute():
        manifest_file = root / manifest_file
    try:
        manifest_rel = str(manifest_file.resolve().relative_to(root_resolved))
    except ValueError:
        manifest_rel = manifest_file.name
    assets = load_assets(manifest_file)
    config["_asset_manifest"] = manifest_rel
    config["_asset_manifest_sha256"] = sha256_file(manifest_file)

    admission = Admission(
        config=config,
        root=root,
        asset_root=asset_root,
        synthetic_root=asset_root,
        source_by_id={pin["id"]: pin for pin in config["source_pins"]},
    )
    for asset in sorted(
        assets,
        key=lambda item: (str(item.get("source_id", "")), str(item.get("asset_id", ""))),
    ):
        admission.admit_one(asset)
    return admission.finish()


def write_outputs(
    root: Path,
    admitted: list[dict],
    rejected: list[dict],
    receipt: dict,
    output_dir: str | Path | None = None,
) -> dict:
    root = Path(root)
    config = receipt.get("config", {})
    config_path = config.get("path", CONFIG_PATH)
    config_file = root / config_path
    if config_file.is_file():
        base_config = json.loads(config_file.read_text(encoding="utf-8"))
        output = base_config.get("outputs", {})
    else:
        output = {}
    directory = Path(output_dir) if output_dir is not None else root
    receipt_path = directory / output.get(
        "receipt", "reports/sparkx_first_tranche_20261026/tranche_receipt.json"
    )
    admitted_path = directory / output.get(
        "admitted", "reports/sparkx_first_tranche_20261026/admitted_assets.jsonl"
    )
    rejected_path = directory / output.get(
        "rejected", "reports/sparkx_first_tranche_20261026/rejected_assets.jsonl"
    )
    for path in (receipt_path, admitted_path, rejected_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    admitted_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in admitted),
        encoding="utf-8",
    )
    rejected_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in rejected),
        encoding="utf-8",
    )
    return {
        "receipt": str(receipt_path),
        "admitted": str(admitted_path),
        "rejected": str(rejected_path),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--assets", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        help="recompute and compare with the committed receipt without writing",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()

    try:
        admitted, rejected, receipt = run(
            root=root,
            config_path=args.config,
            asset_manifest_path=args.assets,
        )
    except (TrancheError, json.JSONDecodeError, OSError) as error:
        print(f"tranche error: {error}", file=sys.stderr)
        return 2

    accepted = (
        receipt["admitted_count"] >= json.loads(
            (root / args.config).read_text(encoding="utf-8")
        )["caps"]["min_admitted_assets"]
        and not receipt["status"].startswith("refused")
    )

    if args.check:
        temporary = root / "reports/sparkx_first_tranche_20261026"
        expected_receipt = temporary / "tranche_receipt.json"
        if not expected_receipt.is_file():
            print(f"receipt missing for --check: {expected_receipt}", file=sys.stderr)
            return 2
        if expected_receipt.read_text(encoding="utf-8") != json.dumps(receipt, indent=2) + "\n":
            print(f"tranche receipt is stale or non-deterministic: {expected_receipt}", file=sys.stderr)
            return 1
        print(f"Tranche receipt verified: {expected_receipt}")
        return 0 if accepted else 1

    if not args.dry_run:
        paths = write_outputs(root, admitted, rejected, receipt, args.output_dir)
        print(
            f"Wrote {paths['receipt']}: {receipt['admitted_count']} admitted, "
            f"{receipt['rejected_count']} rejected, status={receipt['status']}"
        )
    else:
        print(
            f"Dry run: {receipt['admitted_count']} admitted, "
            f"{receipt['rejected_count']} rejected, status={receipt['status']}"
        )
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())

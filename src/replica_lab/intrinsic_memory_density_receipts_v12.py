"""Deterministic receipt generation and no-overwrite output for density v12.

The profile and pilot write their measured JSON through ``write_new_json``.
``write_repair_receipts`` then derives protocol_summary.json and
verification_results.json from those measured files and the validated checked-in
protocols.  No timestamp or host-specific value enters the generated summaries,
so the same inputs and same code reproduce byte-identical receipts.
"""
from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

from .intrinsic_memory_density_protocols_v12 import (
    canonical_digest, load_protocols, summary as protocol_summary,
)

FORMAT = "replica.intrinsic_memory.density.receipts.v12"
REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_SUMMARY = "protocol_summary.json"
VERIFICATION_RESULTS = "verification_results.json"
REPORT_MANIFEST = "report_manifest.json"


_PENDING_CLEANUP = set()
_DURABILITY_MODES = set()


def reset_durability():
    _DURABILITY_MODES.clear()


def _unlink_or_queue(path):
    path = Path(path)
    try:
        path.unlink()
        try:
            _fsync_directory(path.parent)
        except OSError:
            if len(_PENDING_CLEANUP) < 64:
                _PENDING_CLEANUP.add(path)
    except FileNotFoundError:
        return
    except OSError:
        if len(_PENDING_CLEANUP) < 64:
            _PENDING_CLEANUP.add(path)


def _cleanup_pending():
    for path in list(_PENDING_CLEANUP):
        try:
            Path(path).unlink()
            try:
                _fsync_directory(Path(path).parent)
            except OSError:
                break
            _PENDING_CLEANUP.discard(path)
        except FileNotFoundError:
            _PENDING_CLEANUP.discard(path)
        except OSError:
            break


def _portable(path):
    """Repository-relative POSIX identity; no absolute host path."""
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return Path(os.path.relpath(path, REPO_ROOT)).as_posix()


class ReceiptError(ValueError):
    """Receipt input or destination failed closed."""

    def __init__(self, message, *, notes=None):
        super().__init__(message)
        self.notes = list(notes or [])


def durability_label():
    if not _DURABILITY_MODES:
        return "not_executed"
    return "+".join(sorted(_DURABILITY_MODES))


def _durable_fsync(descriptor):
    if sys.platform == "darwin":
        try:
            fcntl.fcntl(descriptor, getattr(fcntl, "F_FULLFSYNC", 51))
            _DURABILITY_MODES.add("darwin_F_FULLFSYNC")
            return "darwin_F_FULLFSYNC"
        except OSError as error:
            if error.errno not in (errno.EINVAL, getattr(errno, "ENOTSUP", errno.EINVAL)):
                raise
    os.fsync(descriptor)
    _DURABILITY_MODES.add("posix_fsync")
    return "posix_fsync"


def _sha256(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ReceiptError(f"receipt input is missing or substituted: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fsync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        _durable_fsync(descriptor)
    finally:
        os.close(descriptor)


def _sync_parent(path):
    try:
        _fsync_directory(Path(path).parent)
        return True
    except OSError:
        return False


def _remove_final(path):
    path = Path(path)
    try:
        path.unlink()
        return _sync_parent(path)
    except FileNotFoundError:
        return True
    except OSError:
        quarantine = path.with_name(f".{path.name}.failed")
        try:
            os.replace(path, quarantine)
            _unlink_or_queue(quarantine)
            return _sync_parent(path)
        except OSError:
            _unlink_or_queue(path)
            return False


def write_new_json(path, value):
    """Atomically create a JSON file; never overwrite prior evidence.

    Content is written and fsynced to a sibling temporary file, then linked to
    the final name with O_EXCL semantics.  A partial write, injected failure or
    concurrent creator leaves no final file and can be retried.
    """
    _cleanup_pending()
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary)
    linked = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            _durable_fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite receipt: {path}")
        linked = True
        try:
            os.unlink(temporary)
        except OSError:
            # The final receipt is complete; bounded cleanup removes this temp
            # name on the next receipt operation without turning success into a
            # raised call that leaves the final path.
            _unlink_or_queue(temporary)
        _fsync_directory(path.parent)
        _cleanup_pending()
    except BaseException as error:
        notes = []
        if linked and not _remove_final(path):
            notes.append(f"final receipt could not be removed: {path}")
        _unlink_or_queue(temporary)
        if notes:
            raise ReceiptError("receipt failure cleanup incomplete", notes=notes) from error
        raise
    return path


def _load(path, label):
    path = Path(path)
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ReceiptError(f"invalid {label} receipt: {path}") from error
    if not isinstance(value, dict):
        raise ReceiptError(f"{label} receipt must be an object")
    return value


def build_protocol_summary(protocols, cost_pilot, cpu_summary, *, repair_cycle):
    if cost_pilot.get("total_cost_protocol_sha256") != canonical_digest(
            protocols["preregistered_measured_total_cost"]):
        raise ReceiptError("cost pilot does not cite the checked-in total-cost protocol")
    if cost_pilot.get("cost_metric") != protocols[
            "preregistered_measured_total_cost"]["pilot"]["cost_metric"]:
        raise ReceiptError("cost pilot metric differs from the frozen protocol")
    if "selected_updates" not in cost_pilot or "matched_arms" not in cost_pilot:
        raise ReceiptError("cost pilot selection receipt is incomplete")
    if cpu_summary.get("protocol_digests") != {
            name: canonical_digest(protocol) for name, protocol in protocols.items()}:
        raise ReceiptError("CPU profile protocol digests differ from checked-in protocols")
    return {
        "format": "replica.intrinsic_memory.density.protocol_report.v12",
        "repair_cycle": repair_cycle,
        "scope": "CPU_protocol_preparation_native_not_admitted",
        "native_execution": False,
        "native_weights_loaded": False,
        "downloads": False,
        "P_updates": 0,
        "promotion": False,
        "protocols": {
            name: {
                "config": f"configs/{filename}",
                "sha256": canonical_digest(protocols[name]),
                "comparison": name,
                "varied_factor": protocols[name]["varied_factor"],
                "confounds": protocols[name]["confounds"],
                "status": protocols[name]["status"],
                "producer_binding": protocols[name].get(
                    "producer_binding", "reviewed_v12_direct"),
            }
            for name, filename in (
                ("fixed_per_site", "fixed_per_site_v12.json"),
                ("matched_unique_parameters", "matched_unique_parameters_v12.json"),
                ("preregistered_measured_total_cost", "measured_total_cost_v12.json"),
            )
        },
        "parameter_accounting": protocol_summary(),
        "cost_pilot": {
            "target_cost_ns": cost_pilot["target_cost_ns"],
            "selected_updates": cost_pilot["selected_updates"],
            "matched_arms": cost_pilot["matched_arms"],
            "unmatched_arms": cost_pilot["unmatched_arms"],
            "total_cost_protocol_sha256": cost_pilot["total_cost_protocol_sha256"],
            "scored_labels_read": cost_pilot.get("scored_labels_read"),
        },
        "post_update_arm_profile": {
            kind: {
                arm: {
                    "logits_sha256": row["post_update"]["logits_sha256"],
                    "logits_changed_from_pre_update": row["post_update"][
                        "logits_changed_from_pre_update"],
                    "reader_invocations": row["post_update"]["reader_invocations"],
                }
                for arm, row in cpu_summary.get(kind, {}).items()
                if row.get("post_update")
            }
            for kind in ("arms", "matched_arms")
        },
        "no_update_control": {
            kind: {
                arm: {
                    "logits_sha256": row["no_update_control"]["logits_sha256"],
                    "unchanged_from_pre_update": row["no_update_control"][
                        "unchanged_from_pre_update"],
                }
                for arm, row in cpu_summary.get(kind, {}).items()
                if row.get("no_update_control")
            }
            for kind in ("arms", "matched_arms")
        },
        "native_consumer_limitation": (
            "No native consumer, disjoint native pilot worlds, end-to-end native "
            "cost measurement or physical-memory qualification exists in this receipt."
        ),
        "claim_limits": [
            "No native density, utility, latency, memory or persistence claim is made.",
            "Parameter estimates exclude producer, order, activations, buffers and backbone.",
            "Random-fixture CPU mechanics are not Spark competence.",
        ],
    }


def build_verification_results(cpu_summary, cost_pilot, protocols, *, repair_cycle):
    if cost_pilot.get("total_cost_protocol_sha256") != canonical_digest(
            protocols["preregistered_measured_total_cost"]):
        raise ReceiptError("cost pilot does not cite the checked-in total-cost protocol")
    if cost_pilot.get("cost_metric") != protocols[
            "preregistered_measured_total_cost"]["pilot"]["cost_metric"]:
        raise ReceiptError("cost pilot metric differs from the frozen protocol")
    for kind in ("arms", "matched_arms"):
        for arm, row in cpu_summary.get(kind, {}).items():
            control = row.get("no_update_control")
            if not control or control.get("unchanged_from_pre_update") is not True:
                raise ReceiptError(f"no-update digest control failed for {kind}/{arm}")
            post = row.get("post_update")
            if not post or post.get("logits_changed_from_pre_update") is not True:
                raise ReceiptError(f"post-update digest control failed for {kind}/{arm}")
            feature = row.get("feature_contract")
            if (not isinstance(feature, dict)
                    or feature.get("input_policy")
                        != "canonical_span_only_no_template_no_special_tokens"
                    or feature.get("output_policy")
                        != "last_decoder_hidden_after_final_norm"
                    or feature.get("original_updates") != 0):
                raise ReceiptError(f"feature contract differs for {kind}/{arm}")
    if len({row["post_update"]["logits_sha256"]
            for row in cpu_summary.get("arms", {}).values()}) < 2:
        raise ReceiptError("fixed post-update profile does not distinguish arms")
    if len({row["post_update"]["logits_sha256"]
            for row in cpu_summary.get("matched_arms", {}).values()}) < 2:
        raise ReceiptError("matched post-update profile does not distinguish arms")
    control = cost_pilot.get("out_of_tolerance_control") or {}
    if (control.get("deliberate_out_of_tolerance_observed") is not True
            or (control.get("arithmetic_control") or {}).get(
                "outside_declared_tolerance") is not True):
        raise ReceiptError("deliberate out-of-tolerance control was not observed")
    if (not control.get("charged_to_arm")
            or type(control.get("control_cost_ns")) is not int
            or control["control_cost_ns"] <= 0):
        raise ReceiptError("out-of-tolerance control cost was not charged")
    counters = {
        name: {
            "sites": row["sites"],
            "physical_reader_groups": row["physical_reader_groups"],
            "reader_invocations": row["reader_invocations"],
            "reader_parameters": row["parameter_audit"]["reader_parameters"],
            "all_intended_parameters": row["parameter_audit"]["intended_parameters"],
            "optimizer_parameter_tensors": row["parameter_audit"][
                "optimizer_parameter_tensors"],
        }
        for name, row in cpu_summary.get("arms", {}).items()
    }
    matched = {
        name: {
            "sites": row["sites"],
            "physical_reader_groups": row["physical_reader_groups"],
            "reader_invocations": row["reader_invocations"],
            "reader_parameters": row["parameter_audit"]["reader_parameters"],
            "reader_state_tensors": row["parameter_audit"]["reader_groups"][
                "state_tensors"],
            "alias_duplicated_state_tensors": row["parameter_audit"]["reader_groups"][
                "alias_duplicated_state_tensors"],
            "feature_contract": row.get("feature_contract"),
        }
        for name, row in cpu_summary.get("matched_arms", {}).items()
    }
    return {
        "format": "replica.intrinsic_memory.density.verification_results.v12." + repair_cycle,
        "repair_cycle": repair_cycle,
        "scope": "CPU_random_36_block_fixture_engineering_only",
        "native_execution": False,
        "native_weights_loaded": False,
        "downloads": False,
        "P_updates": 0,
        "promotion": False,
        "observed_counts": {"fixed_per_site": counters, "matched_v12": matched},
        "post_update_arm_distinguishing_profile": {
            kind: {
                arm: row["post_update"]
                for arm, row in cpu_summary.get(kind, {}).items()
                if row.get("post_update")
            }
            for kind in ("arms", "matched_arms")
        },
        "no_update_unchanged_digest_control": {
            kind: {
                arm: row["no_update_control"]
                for arm, row in cpu_summary.get(kind, {}).items()
                if row.get("no_update_control")
            }
            for kind in ("arms", "matched_arms")
        },
        "fresh_process_continuation": {
            "fixed_v12": cpu_summary.get("fresh_process_continuation"),
            "matched_v12": cpu_summary.get("matched_fresh_process_continuation"),
        },
        "measured_total_cost_pilot": {
            key: cost_pilot.get(key) for key in (
                "cost_metric", "repeat", "aggregate", "declared_common_updates",
                "declared_updates_cap", "target_cost_ns", "selected_updates",
                "matched_arms", "unmatched_arms", "total_cost_protocol_sha256",
                "scored_labels_read", "out_of_tolerance_control",
                "total_charged_cost_ns", "timing_synchronization",
            )
        },
        "protocol_digests": {
            name: canonical_digest(protocol) for name, protocol in protocols.items()},
        "native_consumer_limitation": (
            "CPU-only: no native consumer, no end-to-end native cost or physical "
            "memory measurement, and no disjoint native pilot worlds."
        ),
        "test_evidence": "main-owned density/independent test run after integration",
    }




def _stage_bytes(path, payload):
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            _durable_fsync(stream.fileno())
    except BaseException:
        _unlink_or_queue(temporary)
        raise
    return temporary


def _read_backup(path):
    if not path.exists():
        return None
    raw = path.read_bytes()
    try:
        json.loads(raw)
    except ValueError as error:
        raise ReceiptError(f"corrupt receipt backup: {path}") from error
    return raw


def _stage_raw(path, raw):
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            _durable_fsync(stream.fileno())
    except BaseException:
        _unlink_or_queue(temporary)
        raise
    return temporary


def _restore_backup(path, raw):
    if raw is None:
        return _remove_final(path)
    temporary = _stage_raw(path, raw)
    try:
        os.replace(temporary, path)
    finally:
        _unlink_or_queue(temporary)
    return True


def _transactional_replace(entries):
    """Replace a set of JSON files with best-effort complete rollback."""
    entries = [(Path(path), value) for path, value in entries]
    backups, staged = {}, []
    try:
        for path, value in entries:
            path.parent.mkdir(parents=True, exist_ok=True)
            backups[path] = _read_backup(path)
            payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
            staged.append((path, _stage_bytes(path, payload)))
        for path, temporary in staged:
            os.replace(temporary, path)
        for directory in sorted({path.parent for path, _ in entries}):
            _fsync_directory(directory)
    except BaseException as original:
        notes = []
        for path, backup in backups.items():
            try:
                if not _restore_backup(path, backup):
                    notes.append(f"could not restore {path}")
            except BaseException as error:  # noqa: BLE001 - collected as rollback note
                notes.append(f"restore failed for {path}: {type(error).__name__}")
        for path, temporary in staged:
            _unlink_or_queue(temporary)
        for directory in sorted({path.parent for path, _ in entries}):
            try:
                _fsync_directory(directory)
            except OSError:
                notes.append(f"directory fsync failed after rollback: {directory}")
        if notes:
            raise ReceiptError(
                "receipt transaction rollback incomplete", notes=notes) from original
        raise


def _write_replace_json(path, value):
    """Backward-compatible single-file atomic replacement."""
    _transactional_replace([(path, value)])


def read_current_navigation(root):
    """Read CURRENT only, verify every canonical hash, otherwise fail closed.

    CURRENT is the sole navigation authority.  NONCANONICAL markers are never
    consulted for recovery; a missing, corrupt or hash-mismatched CURRENT is an
    explicit restart-resolution failure.
    """
    root = Path(root)
    current_path = root / "CURRENT.json"
    if current_path.is_symlink() or not current_path.is_file():
        raise ReceiptError("CURRENT navigation receipt is missing or substituted")
    current = _load(current_path, "current navigation")
    if current.get("format") != "replica.intrinsic_memory.density.current_navigation.v12":
        raise ReceiptError("CURRENT navigation format differs")
    destination = current.get("canonical_destination")
    files = current.get("files")
    if not isinstance(destination, str) or not destination or not isinstance(files, dict):
        raise ReceiptError("CURRENT navigation is incomplete")
    canonical = Path(destination)
    if not canonical.is_absolute():
        canonical = (REPO_ROOT / canonical).resolve()
    if not canonical.is_dir():
        raise ReceiptError("CURRENT canonical directory is missing")
    required = ("cpu_engineering_summary.json", "cost_pilot_summary.json",
                "protocol_summary.json", "verification_results.json",
                "report_manifest.json")
    if set(files) != set(required):
        raise ReceiptError("CURRENT canonical file schema differs")
    for name in required:
        if _sha256(canonical / name) != files[name]:
            raise ReceiptError(f"CURRENT hash mismatch for {name}")
    return current


def write_current_navigation(root, canonical_directory, *, generator,
                             x6_test):
    """Generate CURRENT/NONCANONICAL consistently from measured canonical files."""
    root = Path(root)
    canonical = Path(canonical_directory)
    if not canonical.is_absolute():
        canonical = (canonical.resolve() if canonical.is_dir()
                     else (root / canonical).resolve())
    else:
        canonical = canonical.resolve()
    if not canonical.is_dir():
        raise ReceiptError(f"canonical directory is missing: {canonical}")
    required = ("cpu_engineering_summary.json", "cost_pilot_summary.json",
                "protocol_summary.json", "verification_results.json",
                "report_manifest.json")
    missing = [name for name in required if not (canonical / name).is_file()]
    if missing:
        raise ReceiptError(
            "canonical receipt schema filenames are required: " + ", ".join(missing))
    files = {name: _sha256(canonical / name) for name in required}
    superseded = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.resolve() == canonical:
            continue
        if (path / "cost_pilot_summary.json").is_file() or (
                path / "cpu_engineering_summary.json").is_file():
            superseded.append(path)
    canonical_name = _portable(canonical)
    root_name = _portable(root)
    noncanonical = {
        "format": "replica.intrinsic_memory.density.noncanonical.v12",
        "canonical_destination": canonical_name,
        "generated_by": generator,
        "noncanonical_destinations": [
            {"path": root_name,
             "reason": "historical root receipts; bytes preserved; superseded by canonical receipt"},
            *[
                {"path": _portable(path),
                 "reason": "historical bytes preserved; superseded by canonical receipt"}
                for path in superseded
            ],
        ],
        "files_unchanged": True,
        "native_execution": False,
    }
    entries = [(root / "NONCANONICAL.json", noncanonical)]
    for path in superseded:
        entries.append((path / "NONCANONICAL.json", {
            "format": "replica.intrinsic_memory.density.noncanonical.v12",
            "canonical_destination": canonical_name,
            "generated_by": generator,
            "reason": "historical receipt bytes preserved; noncanonical",
        }))
    current = {
        "format": "replica.intrinsic_memory.density.current_navigation.v12",
        "canonical_destination": canonical_name,
        "files": files,
        "generator": generator,
        "x6_test": x6_test,
        "native_execution": False,
        "P_updates": 0,
    }
    entries.append((root / "CURRENT.json", current))
    try:
        _transactional_replace(entries)
    except ReceiptError as error:
        if error.notes:
            # An incomplete rollback must leave no mixed authority set.
            for path, _ in entries:
                if not _remove_final(path):
                    error.notes.append(f"could not remove ambiguous navigation file: {path}")
        raise
    return read_current_navigation(root)


def write_repair_receipts(directory, *, cpu_summary_path, cost_pilot_path,
                          protocols_directory="configs", repair_cycle=None):
    reset_durability()
    directory = Path(directory)
    derived_cycle = directory.resolve().name
    if repair_cycle is None:
        repair_cycle = derived_cycle
    elif repair_cycle != derived_cycle:
        raise ReceiptError(
            "repair_cycle must match the canonical directory identity: "
            f"{repair_cycle!r} != {derived_cycle!r}")
    cpu_path, cost_path = Path(cpu_summary_path), Path(cost_pilot_path)
    cpu_summary = _load(cpu_path, "CPU profile")
    cost_pilot = _load(cost_path, "cost pilot")
    protocols = load_protocols(protocols_directory)
    protocol_summary_value = build_protocol_summary(
        protocols, cost_pilot, cpu_summary, repair_cycle=repair_cycle)
    verification = build_verification_results(
        cpu_summary, cost_pilot, protocols, repair_cycle=repair_cycle)
    created = []
    try:
        write_new_json(directory / PROTOCOL_SUMMARY, protocol_summary_value)
        created.append(directory / PROTOCOL_SUMMARY)
        write_new_json(directory / VERIFICATION_RESULTS, verification)
        created.append(directory / VERIFICATION_RESULTS)
        manifest = {
            "format": FORMAT,
            "repair_cycle": repair_cycle,
            "inputs": {
                "cpu_summary": {"path": _portable(cpu_path),
                                "sha256": _sha256(cpu_path)},
                "cost_pilot": {"path": _portable(cost_path),
                               "sha256": _sha256(cost_path)},
                "protocols_directory": _portable(protocols_directory),
            },
            "outputs": {
                PROTOCOL_SUMMARY: _sha256(directory / PROTOCOL_SUMMARY),
                VERIFICATION_RESULTS: _sha256(directory / VERIFICATION_RESULTS),
            },
            "native_execution": False,
            "durability": durability_label(),
            "deterministic": True,
        }
        write_new_json(directory / REPORT_MANIFEST, manifest)
        created.append(directory / REPORT_MANIFEST)
        return manifest
    except BaseException as error:
        # This call's partial receipt set must not block a clean retry.
        notes = []
        for path in created:
            if not _remove_final(path):
                notes.append(f"could not remove partial receipt: {path}")
        if notes:
            raise ReceiptError(
                "receipt set rollback incomplete", notes=notes) from error
        raise

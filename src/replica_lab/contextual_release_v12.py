"""Immutable producer-release overlays over existing canonical evidence authority.

A release never rewrites E records or historical values. It re-encodes current,
scoped canonical sources and atomically publishes a digest-bound generation.
Interrupted preparation leaves the previous committed generation usable. Readers
refuse a different/mixed release and revalidate original E authority at each site.
"""
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import torch

from .contextual_feature_identity_v12 import (
    CanonicalSpan, ContextualMemoryError, CODEC, PRODUCER, REPRESENTATION)
from .contextual_memory_v12 import atomic_json, digest, file_hash
from .tiered_memory import (
    _read_verified, pack_values, unpack_values, RetrievalBudget, MemoryScope)

FORMAT = "replica.contextual_producer_release.v12"


class WorkLedgerReleaseReadAccounting:
    """Map release-corpus direct reads onto a native ``artifact_read_bytes`` ledger."""

    def __init__(self, ledger, stage):
        if not callable(getattr(ledger, "reserve", None)) or not callable(getattr(ledger, "finish", None)):
            raise ContextualMemoryError("explicit read ledger with reserve/finish required")
        if not isinstance(stage, str) or not stage:
            raise ContextualMemoryError("read-ledger stage required")
        self.ledger, self.stage = ledger, stage

    def reserve_read(self, operation, byte_count, file_count=1):
        if (not isinstance(operation, str) or not operation or type(byte_count) is not int
                or byte_count < 0 or type(file_count) is not int or file_count < 1):
            raise ContextualMemoryError("invalid release read reservation")
        return self.ledger.reserve(self.stage, operation, artifact_read_bytes=byte_count)

    def complete_read(self, token, *, actual_bytes, status="completed"):
        if status not in ("completed", "failed"):
            raise ContextualMemoryError("invalid release read status")
        actual = {"artifact_read_bytes": actual_bytes} if status == "completed" else {}
        return self.ledger.finish(token, status=status, actual=actual)


class ProducerReleaseCorpus:
    def __init__(self, store, scope, *, max_records=1024, read_accounting=None):
        if not isinstance(scope, MemoryScope):
            raise ContextualMemoryError("a validated memory scope is required")
        if type(max_records) is not int or not 1 <= max_records <= 1024:
            raise ContextualMemoryError("release corpus bound must be 1..1024 records")
        if read_accounting is not None and (
                not callable(getattr(read_accounting, "reserve_read", None))
                or not callable(getattr(read_accounting, "complete_read", None))):
            raise ContextualMemoryError("release read accounting must reserve/complete reads")
        self.store, self.scope, self.max_records = store, scope, max_records
        self.read_accounting = read_accounting
        self.root = store.root / "artifacts" / "contextual_v12" / "releases" / scope.scope_key
        self.root.mkdir(parents=True, exist_ok=True)
        self.cost = dict(source_read_bytes=0, source_read_files=0, write_bytes=0,
                         write_ns=0, read_bytes=0, read_files=0, publications=0,
                         authority_read_bytes=0, authority_read_files=0,
                         artifact_read_bytes=0, artifact_read_files=0)
        # Generation zero is explicit authority, so an interrupted first build
        # can resume without confusing a lost committed pointer with an empty
        # corpus. Never reconstruct authority from whichever payload survived.
        lock = self.root / "writer.lock"
        if lock.is_symlink():
            raise ContextualMemoryError("substituted release lock")
        with lock.open("a") as locked:
            fcntl.flock(locked, fcntl.LOCK_EX | fcntl.LOCK_NB)
            pointer = self.root / "current.json"
            if not pointer.exists():
                if any(p.is_dir() and p.name.isdigit() for p in self.root.iterdir()):
                    raise ContextualMemoryError("committed producer pointer missing; recovery required")
                origin = {"format": FORMAT + ".pointer", "scope": asdict(scope),
                          "generation": 0, "manifest_sha256": None}
                size = len(json.dumps(origin, sort_keys=True, indent=2).encode()) + 1
                component = "release_pointer:" + scope.scope_key
                store._charge("cold", "artifacts", size, component, "release origin")
                try:
                    atomic_json(pointer, origin)
                except BaseException as error:
                    try:
                        store._release(component)
                    except BaseException as release_error:
                        error.add_note(f"origin charge rollback failed: {release_error}")
                        raise ContextualMemoryError(
                            "release origin charge rollback failed") from error
                    raise
                self.cost["write_bytes"] += size
        self._manifest_cache = None

    # ------------------------------------------------------------------
    # Authority reads

    def _reserve_read(self, operation, byte_count):
        if self.read_accounting is None:
            return None
        return self.read_accounting.reserve_read(operation, byte_count, 1)

    def _complete_read(self, token, *, actual_bytes, status="completed"):
        if token is not None:
            self.read_accounting.complete_read(token, actual_bytes=actual_bytes, status=status)

    def _read_direct(self, operation, path):
        size = path.stat().st_size
        token = self._reserve_read(operation, size)
        try:
            payload = path.read_bytes()
        except BaseException:
            self._complete_read(token, actual_bytes=0, status="failed")
            raise
        self._complete_read(token, actual_bytes=len(payload))
        self.cost["artifact_read_bytes"] += len(payload)
        self.cost["artifact_read_files"] += 1
        return payload

    def _authority(self, path):
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 4 * 1024**2:
            raise ContextualMemoryError("missing/substituted/oversized release authority")
        payload = self._read_direct("release_authority", path)
        self.cost["authority_read_bytes"] += len(payload)
        self.cost["authority_read_files"] += 1
        return hashlib.sha256(payload).hexdigest(), payload

    def _write(self, path, payload):
        component = "release_v12:" + str(path.relative_to(self.store.root))
        self.store._charge("cold", "artifacts", len(payload), component,
                           "immutable producer release")
        start = time.monotonic_ns()
        created = False
        try:
            with path.open("xb") as out:
                created = True
                out.write(payload)
                out.flush()
                os.fsync(out.fileno())
        except BaseException:
            # A failed immutable write is not durable work: remove any partial
            # file and release the reservation so a retry is not double-charged.
            if created:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            self.store._release(component)
            raise
        self.cost["write_bytes"] += len(payload)
        self.cost["write_ns"] += time.monotonic_ns() - start

    def _current(self):
        pointer = self.root / "current.json"
        if not pointer.exists():
            raise ContextualMemoryError("producer pointer missing")
        _, payload = self._authority(pointer)
        pin = json.loads(payload)
        if (pin.get("format") != FORMAT + ".pointer"
                or MemoryScope(**pin["scope"]).scope_key != self.scope.scope_key):
            raise ContextualMemoryError("release pointer scope/format differs")
        generation = pin.get("generation")
        if type(generation) is int and generation == 0 and pin.get("manifest_sha256") is None:
            return None
        if type(generation) is not int or generation < 1:
            raise ContextualMemoryError("release generation invalid")
        if type(pin.get("e_epoch")) is not int or pin["e_epoch"] < 0:
            raise ContextualMemoryError("release pointer has no E authority sequence")
        path = self.root / f"{generation:08d}" / "manifest.json"
        identity, payload = self._authority(path)
        if identity != pin.get("manifest_sha256"):
            raise ContextualMemoryError("committed producer authority missing/corrupt")
        if self._manifest_cache is None or self._manifest_cache[0] != identity:
            m = json.loads(payload)
            if (m.get("format") != FORMAT or m.get("generation") != generation
                    or MemoryScope(**m["scope"]).scope_key != self.scope.scope_key
                    or m.get("representation") != REPRESENTATION
                    or m.get("codec") != CODEC
                    or m.get("record_digest") != digest(m.get("records"))):
                raise ContextualMemoryError("release manifest identity differs")
            if any(r["producer_release"] != m["producer_release"] for r in m["records"].values()):
                raise ContextualMemoryError("mixed producer releases refused")
            self._manifest_cache = (identity, m)
        return pin, self._manifest_cache[1]

    # ------------------------------------------------------------------
    # Source pins

    def _source(self, row):
        source = None
        for item in json.loads(row["source_lineage"]):
            try:
                value = json.loads(item)
            except (ValueError, TypeError):
                continue
            if isinstance(value, dict) and value.get("record_id") == row["record_id"]:
                source = value
        if source is None:
            raise ContextualMemoryError("canonical provenance missing")
        text_bytes = int(row["text_bytes"])
        token = self._reserve_read("release_source", text_bytes)
        try:
            text = _read_verified(row["text_path"], self.store.root, row["text_sha256"],
                                  text_bytes, accounting=None).decode()
        except BaseException:
            self._complete_read(token, actual_bytes=0, status="failed")
            raise
        self._complete_read(token, actual_bytes=text_bytes)
        self.cost["source_read_bytes"] += text_bytes
        self.cost["source_read_files"] += 1
        self.cost["artifact_read_bytes"] += text_bytes
        self.cost["artifact_read_files"] += 1
        data = {**source, "source_id": row["source_id"], "text": text}
        span = CanonicalSpan(**{k: data[k] for k in CanonicalSpan.__dataclass_fields__}).validate()
        pin = {"record_id": row["record_id"], "cluster_id": row["cluster_id"],
               "generation": row["cluster_generation"], "revision": row["revision"],
               "text_sha256": row["text_sha256"]}
        return span, pin, source

    def _validate_source_pin(self, pin):
        row = self.store.connection.execute(
            "SELECT r.*,c.current_generation FROM records r JOIN clusters c "
            "ON r.cluster_id=c.cluster_id WHERE r.record_id=?", (pin["record_id"],)).fetchone()
        cutoff = self.scope.cutoff
        if (row is None or row["status"] != "current" or row["scope_key"] != self.scope.scope_key
                or row["revision"] != pin["revision"] or row["text_sha256"] != pin["text_sha256"]
                or row["current_generation"] != pin["generation"] or row["observed_at"] > cutoff
                or (row["valid_from"] is not None and row["valid_from"] > cutoff)
                or (row["valid_to"] is not None and row["valid_to"] <= cutoff)):
            raise ContextualMemoryError("corrected/revoked/time/generation source invalidated release")
        return True

    def _payload_path(self, manifest, record):
        name = record.get("value_file")
        if (not isinstance(name, str) or len(name) != 70 or not name.endswith(".slots")
                or any(character not in "0123456789abcdef" for character in name[:64])):
            raise ContextualMemoryError("release payload identity differs")
        path = self.root / f"{manifest['generation']:08d}" / name
        if path.is_symlink() or not path.is_file():
            raise ContextualMemoryError("substituted release payload")
        return path

    # ------------------------------------------------------------------
    # Pointer publication

    @staticmethod
    def _fsync_directory(directory):
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _stage_pointer(self, pointer):
        current = self.root / "current.json"
        pending = current.with_name(current.name + ".pending." + uuid.uuid4().hex)
        atomic_json(pending, pointer)
        return pending

    def _restore_pointer(self, previous):
        current = self.root / "current.json"
        if previous is None:
            try:
                current.unlink()
            except FileNotFoundError:
                pass
        else:
            temporary = current.with_name(current.name + ".restore." + uuid.uuid4().hex)
            with temporary.open("xb") as stream:
                stream.write(previous)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, current)
        try:
            self._fsync_directory(current.parent)
        except BaseException:
            if previous is not None:
                try:
                    if current.read_bytes() != previous:
                        raise
                except FileNotFoundError:
                    raise
            # With the previous bytes live (or no authority) the rollback is
            # authority-safe; the original publication error remains primary.
            return

    def _publish_pointer(self, pending, previous):
        current = self.root / "current.json"
        replaced = False
        try:
            os.replace(pending, current)
            replaced = True
            self._fsync_directory(current.parent)
        except BaseException as error:
            if replaced:
                try:
                    self._restore_pointer(previous)
                except BaseException as restore_error:
                    # Cannot restore the previous pointer: remove authority so
                    # no failed new generation can be consumed.
                    error.add_note(f"pointer restore failed: {restore_error}")
                    try:
                        current.unlink()
                        self._fsync_directory(current.parent)
                    except BaseException:
                        raise ContextualMemoryError(
                            "release pointer rollback failed; authority state uncertain"
                        ) from error
                    raise ContextualMemoryError(
                        "release pointer rollback failed; release authority removed"
                    ) from error
            raise
        finally:
            try:
                pending.unlink()
            except FileNotFoundError:
                pass

    def _revert_pointer_charge(self, previous):
        component = "release_pointer:" + self.scope.scope_key
        old_size = 0 if previous is None else len(previous)
        with self.store._transaction():
            self.store._set_component_locked(component, "cold", "artifacts", old_size,
                                             "release authority")

    def rebuild(self, path, *, before_publish=None):
        """One explicit release transaction; hook supports portable crash testing."""
        if not isinstance(getattr(path, "source_identity", None), str) or path.source_identity == "":
            raise ContextualMemoryError("source identity missing")
        if not callable(getattr(path.encoder.provider, "clear", None)):
            raise ContextualMemoryError("source feature provider required")
        lock = self.root / "writer.lock"
        if lock.is_symlink():
            raise ContextualMemoryError("substituted release lock")
        with lock.open("a") as locked:
            fcntl.flock(locked, fcntl.LOCK_EX | fcntl.LOCK_NB)
            old = self._current()
            # Orphaned preparation directories remain preserved and are never reused.
            existing = [int(p.name) for p in self.root.iterdir() if p.is_dir() and p.name.isdigit()]
            generation = max(existing, default=0) + 1
            dest = self.root / f"{generation:08d}"
            dest.mkdir()
            version = path.encoder.release()
            rows = self.store._eligible_record_rows(self.scope)
            if not rows or len(rows) > self.max_records:
                raise ContextualMemoryError("release corpus bound/empty")
            records = {}
            for row in rows:
                span, pin, source = self._source(row)
                if path.provenance(span, source.get("producer_revision")) != source:
                    raise ContextualMemoryError("canonical provenance differs")
                with torch.no_grad():
                    values = path._encode([span])[0].detach().cpu()
                payload = b"".join(pack_values(values.tolist()))
                name = hashlib.sha256(span.record_id.encode()).hexdigest() + ".slots"
                self._write(dest / name, payload)
                records[span.record_id] = {"source_pin": pin, "source_provenance": source,
                    "span": asdict(span), "producer_release": version, "value_file": name,
                    "value_bytes": len(payload), "value_sha256": hashlib.sha256(payload).hexdigest()}
            if path.encoder.release() != version:
                raise ContextualMemoryError("producer changed during rebuild")
            m = {"format": FORMAT, "generation": generation, "scope": asdict(self.scope),
                 "parent": old[0]["generation"] if old else None, "producer_release": version,
                 "source_identity": path.source_identity, "representation": REPRESENTATION,
                 "codec": CODEC, "records": records, "record_digest": digest(records)}
            self._write(dest / "manifest.json",
                        (json.dumps(m, sort_keys=True, indent=2) + "\n").encode())
            fd = os.open(dest, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            if before_publish is not None:
                before_publish()
            current_pointer = self.root / "current.json"
            previous_pointer = (current_pointer.read_bytes()
                                if current_pointer.is_file() and not current_pointer.is_symlink()
                                else None)
            pending = None
            try:
                # E's SQLite write lock closes the authority-change race. The
                # staged pointer is durable but not live until after commit.
                with self.store._transaction():
                    if path.encoder.release() != version:
                        raise ContextualMemoryError("producer changed before publication")
                    for record in records.values():
                        self._validate_source_pin(record["source_pin"])
                    pointer = {"format": FORMAT + ".pointer", "scope": asdict(self.scope),
                               "generation": generation,
                               "manifest_sha256": file_hash(dest / "manifest.json"),
                               "e_epoch": int(self.store.epoch)}
                    pointer_bytes = len(json.dumps(pointer, sort_keys=True, indent=2).encode()) + 1
                    self.store._ensure_capacity_locked("cold", "artifacts", pointer_bytes * 2)
                    self.store._set_component_locked(
                        "release_pointer:" + self.scope.scope_key, "cold", "artifacts",
                        pointer_bytes, "release authority")
                    pending = self._stage_pointer(pointer)
            except BaseException:
                if pending is not None:
                    try:
                        pending.unlink()
                    except FileNotFoundError:
                        pass
                raise
            try:
                # Commit and accounting are complete; cache cleanup and the
                # atomic pointer replacement are the only remaining steps. Any
                # failure rolls the charge back; a post-replace failure also
                # restores the previous pointer.
                path.encoder.provider.clear()
                self._publish_pointer(pending, previous_pointer)
            except BaseException as error:
                try:
                    pending.unlink()
                except FileNotFoundError:
                    pass
                try:
                    self._revert_pointer_charge(previous_pointer)
                except BaseException as charge_error:
                    error.add_note(f"pointer charge rollback failed: {charge_error}")
                    raise ContextualMemoryError(
                        "release pointer charge rollback failed") from error
                raise
            self.cost["write_bytes"] += pointer_bytes
            self.cost["publications"] += 1
            return m

    # ------------------------------------------------------------------
    # Consumption

    def values(self, pin, span, *, version, scope, source_identity, payload=None):
        if scope != self.scope:
            raise ContextualMemoryError("release scope/time differs")
        current = self._current()
        if current is None:
            raise ContextualMemoryError("no committed producer release")
        pointer, m = current
        if m.get("source_identity") != source_identity:
            raise ContextualMemoryError("release source execution identity differs")
        record = m["records"].get(span.record_id)
        if (m["producer_release"] != version or record is None
                or record["source_pin"] != pin or record["span"] != asdict(span)):
            raise ContextualMemoryError("missing/mixed/stale producer release corpus")
        self._validate_source_pin(pin)
        path = self._payload_path(m, record)
        if payload is None:
            payload = self._read_direct("release_payload", path)
            self.cost["read_bytes"] += len(payload)
            self.cost["read_files"] += 1
        else:
            payload = bytes(payload)
        if (len(payload) != record["value_bytes"]
                or hashlib.sha256(payload).hexdigest() != record["value_sha256"]):
            raise ContextualMemoryError("release value digest differs")
        values = torch.tensor(unpack_values(
            tuple(payload[i:i + 1024] for i in range(0, len(payload), 1024))))
        if tuple(values.shape) != (4, 256) or not torch.isfinite(values).all():
            raise ContextualMemoryError("release value shape differs")
        return values, {"producer_generation": m["generation"],
                        "producer_manifest": pointer["manifest_sha256"]}

    def validate_batch(self, batch, *, scope, source_identity):
        # Live training sources may originate from an older release. Their E
        # authority is still checked; inference must bind the published release.
        if batch.training:
            return True
        current = self._current()
        if current is None:
            raise ContextualMemoryError("producer authority unavailable")
        pointer, m = current
        if (scope != self.scope or m.get("source_identity") != source_identity
                or batch.producer_version != m["producer_release"]):
            raise ContextualMemoryError("mixed producer release batch")
        if len(batch.pins) != len(batch.spans):
            raise ContextualMemoryError("complete release pins required")
        for pin in batch.pins:
            if (pin.get("producer_generation") != m["generation"]
                    or pin.get("producer_manifest") != pointer["manifest_sha256"]):
                raise ContextualMemoryError("producer generation invalidated batch")
            self._validate_source_pin(pin)
        return True

    def read(self, path, question, *, scope, wait_seconds=0.0):
        result = self.store.select(question, scope=scope,
                                   budget=RetrievalBudget(max_records=2, max_value_slots=8),
                                   include_text=False, admit_payload=False)
        path.cost["retrieval_calls"] += 1
        path.cost["read_file_bytes"] += result.io_bytes
        path.cost["read_file_calls"] += result.file_reads
        try:
            if result.truncated or not result.outcome.ok:
                raise ContextualMemoryError("incomplete release selection")
            current = self._current()
            if current is None:
                raise ContextualMemoryError("no committed producer release")
            pointer, m = current
            if m["producer_release"] != path.encoder.release():
                raise ContextualMemoryError("stale producer release corpus")
            if m.get("source_identity") != path.source_identity:
                raise ContextualMemoryError("release source execution identity differs")
            records, payloads = [], []
            for r in result.records:
                entry = m["records"].get(r.record_id)
                if entry is None:
                    raise ContextualMemoryError("unrebuilt source in release selection")
                span = CanonicalSpan(**entry["span"]).validate()
                pin = entry["source_pin"]
                self._validate_source_pin(pin)
                records.append({"record_id": r.record_id, "text": span.text, "values": None,
                    "source_id": span.source_id, "source_epoch": span.source_epoch,
                    "source_provenance": entry["source_provenance"],
                    "value_producer_id": PRODUCER,
                    "value_producer_version": m["producer_release"],
                    "scope_key": scope.scope_key, "text_sha256": pin["text_sha256"],
                    "supersedes": span.supersedes, "cluster_id": pin["cluster_id"],
                    "cluster_generation": pin["generation"], "revision": pin["revision"]})
                payload_path = self._payload_path(m, entry)
                payloads.append({"record_id": r.record_id,
                    "path": str(payload_path.relative_to(self.store.root)),
                    "sha256": entry["value_sha256"], "bytes": entry["value_bytes"]})
            if not records:
                raise ContextualMemoryError("no eligible release sources")
            # Release artifacts are disk-backed, regardless of E's placement.
            # Route these through the same bounded pending API, never a hidden
            # synchronous cold payload read at a decoder injection site.
            return path.begin_release_read(self.store, scope, records, payloads,
                                           pointer, wait_seconds=wait_seconds)
        finally:
            self.store.release_selection(result.selection_id)

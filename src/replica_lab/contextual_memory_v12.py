"""V12 source-feature memory; v5 durable ticket semantics under a new namespace.

The copied lifecycle preserves sealed v5 bytes. Source/representation/codec and
checkpoint identities are new; no old value or checkpoint is implicitly upgraded.
Live training values stay differentiable. Durable reads, tickets and checkpoints
detach. Every active reader revalidates authority at its injection boundary.
"""
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

import torch

from .contextual_feature_identity_v12 import (
    AuthorityRejection, CanonicalSpan, ContextualMemoryError, ValueBatch, PRODUCER, REPRESENTATION, CODEC,
    SLOTS, WIDTH, tensor_digest, lineage_json)
from .contextual_memory_layout_v12 import ContextualMemoryLayoutV12
from .contextual_memory_v5 import (
    PendingContextualRead, AuthorityReadFailure, ContextualReadFailure, atomic_json, digest, file_hash,
    effective_scope, _release_payloads, _descriptor)
from .tiered_memory import (
    MemoryScope, RecallTicket, RecallResult, RecalledRecord, Outcome,
    EvidenceRecordDraft, _read_verified, unpack_values)
from .tiered_readout import ReadoutBudget, read_scoped_memory, resume_scoped_readout

FORMAT = "replica.contextual_memory.v12"
SLOT_POLICY = "fixed_eight_masked_padding_v12"
MAX_RECORDS = 2

# Kept as a distinct module global so the public ``_read_verified`` name remains
# patchable for accounting tests without charging the worker's real payload
# reads twice.
_payload_read_verified = _read_verified


def _cold_payloads(root, tickets):
    """Stage cold text/values with accounting, returning per-record failures.

    Every actual payload read is charged through ``accounting``. A corrupt or
    missing record is marked invalid and the worker continues so healthy
    siblings remain staged and their bounded C can be preserved.
    """
    accounting = {"file_reads": 0, "io_bytes": 0}
    staged = {}
    invalid = {}
    for ticket in tickets:
        for pin in ticket["pinned"]:
            try:
                text = _payload_read_verified(
                    pin["text_path"], root, pin["text_sha256"], pin["text_bytes"],
                    accounting=accounting)
                values = tuple(
                    _payload_read_verified(value["path"], root, value["sha256"],
                                           value["bytes"], accounting=accounting)
                    for value in pin["values"])
            except Exception as error:
                invalid[pin["record_id"]] = error
                continue
            staged[pin["record_id"]] = (text, values)
    if invalid:
        return staged, accounting, invalid, invalid
    return staged, accounting, None, set()


class ContextualRecordError(ContextualMemoryError):
    """A single durable record failed authority/version validation."""

    def __init__(self, reason, record_id=None):
        super().__init__(reason)
        self.record_id = record_id


class ContextualMemoryV12(ContextualMemoryLayoutV12):
    def __init__(self, *args, objective_identity=None,
                 execution_identity="CPU_engineering_unbound", **kwargs):
        super().__init__(*args, **kwargs)
        self.objective_identity = objective_identity or {"scope": "CPU_correctness_only"}
        if not isinstance(execution_identity, str) or not execution_identity:
            raise ContextualMemoryError("explicit execution identity required")
        self.execution_identity = execution_identity
        self._release_original = self.representation_release
        self._release_cache = None
        self.encoder.release = self._producer_release
        # Fused AdamW does not necessarily advance Tensor._version. Invalidate
        # before its work (also on a partially failed update), not only by stamps.
        self._optimizer_release_hook = self.optimizer.register_step_pre_hook(self._before_step)
        self._pool = None
        self._jobs = {}
        self.release_corpus = None
        self.cost.update({key: 0 for key in (
            "producer_hash_calls", "producer_hash_ns", "producer_hash_cache_hits",
            "authority_validation_ns", "cold_wait_ns", "cold_reserved_bytes",
            "cold_reserved_files", "cold_completed_bytes", "cold_completed_files",
            "durable_write_bytes", "durable_write_ns", "durable_artifact_growth_bytes",
            "admitted_payload_bytes", "ticket_descriptor_read_bytes",
            "ticket_descriptor_read_calls")})

    # ------------------------------------------------------------------
    # Producer release identity

    def _before_step(self, optimizer, args, kwargs):
        self._release_cache = None

    def _producer_release(self):
        stamp = (self.feature_provider.contract_digest(),
                 tuple((name, parameter._version, parameter.data_ptr(), str(parameter.device),
                        str(parameter.dtype), tuple(parameter.shape))
                       for module in self.modules().values()
                       for name, parameter in module.named_parameters()))
        if self._release_cache is not None and self._release_cache[0] == stamp:
            self.cost["producer_hash_cache_hits"] += 1
            return self._release_cache[1]
        start = time.monotonic_ns()
        release = self._release_original()
        self.cost["producer_hash_calls"] += 1
        self.cost["producer_hash_ns"] += time.monotonic_ns() - start
        self._release_cache = (stamp, release)
        return release

    def training_values(self, spans, *, scope):
        return super().training_values(spans, scope=effective_scope(scope))

    # ------------------------------------------------------------------
    # Durable admission and reading

    def admit(self, store, span, *, scope, cluster="contextual", placement="warm"):
        span.validate()
        if (span.available_at > scope.cutoff
                or (span.valid_until is not None and scope.cutoff >= span.valid_until)):
            raise ContextualMemoryError("future/expired source admission")
        version = self.encoder.release()
        with torch.no_grad():
            values = self._encode([span])[0].detach().cpu().tolist()
        draft = EvidenceRecordDraft(span.record_id, span.text, span.source_id, span.source_epoch,
            span.available_at, valid_to=span.valid_until, authenticity="source_report",
            verification_state="verified_source", values=tuple(tuple(v) for v in values),
            value_producer_id=PRODUCER, value_producer_version=version,
            source_lineage=(lineage_json(self.provenance(span, version)),))
        start = time.monotonic_ns()
        before = sum(p.stat().st_size for p in store.root.rglob("*") if p.is_file())
        self.cost["admission_calls"] += 1
        self.cost["admitted_payload_bytes"] += len(span.text.encode()) + SLOTS * WIDTH * 4
        try:
            result = store.admit_cluster(cluster, scope=scope, records=[draft], placement=placement)
        finally:
            self.cost["durable_write_ns"] += time.monotonic_ns() - start
            self.cost["durable_artifact_growth_bytes"] += max(
                0, sum(p.stat().st_size for p in store.root.rglob("*") if p.is_file()) - before)
        return result

    def validate(self, batch, scope):
        if not isinstance(batch, ValueBatch):
            raise ContextualMemoryError("complete ValueBatch required; pending is not usable memory")
        start = time.monotonic_ns()
        try:
            super().validate(batch, effective_scope(scope))
            if (not batch.training and self.release_corpus is None
                    and any("producer_generation" in pin for pin in batch.pins)):
                raise ContextualMemoryError(
                    "published release authority is required for durable overlay values")
            if self.release_corpus is not None:
                self.release_corpus.validate_batch(
                    batch, scope=scope, source_identity=self.source_identity)
        finally:
            self.cost["authority_validation_ns"] += time.monotonic_ns() - start
        return True

    def _batch(self, records, store, scope):
        version = self.encoder.release()
        ordered = sorted(records, key=lambda r: (r["source_id"], r["source_epoch"],
                                                r["source_provenance"].get("source_order", 0),
                                                r["record_id"]))
        if not ordered or len(ordered) > MAX_RECORDS:
            raise ContextualMemoryError("bounded two-span/nonempty selection required")
        if len({r["record_id"] for r in ordered}) != len(ordered):
            raise ContextualMemoryError("duplicate record in one contextual selection")
        spans, values, pins = [], [], []
        for r in ordered:
            record_id = r["record_id"]
            p = r["source_provenance"]
            if (r["scope_key"] != scope.scope_key or r["value_producer_id"] != PRODUCER
                    or p.get("record_id") != record_id or p.get("text_sha256") != r["text_sha256"]
                    or p.get("supersedes") != r["supersedes"]
                    or p.get("representation_version") != REPRESENTATION
                    or p.get("codec_version") != CODEC):
                raise ContextualRecordError("canonical value provenance differs", record_id)
            metadata = {**p, "text": r["text"], "source_id": r["source_id"]}
            span = CanonicalSpan(**{k: metadata[k] for k in CanonicalSpan.__dataclass_fields__}).validate()
            if self.provenance(span, p.get("producer_revision")) != p:
                raise ContextualRecordError("slot provenance differs", record_id)
            pin = {"record_id": record_id, "cluster_id": r["cluster_id"],
                   "generation": r["cluster_generation"], "revision": r["revision"],
                   "text_sha256": r["text_sha256"]}
            if self.release_corpus is None:
                if (r["value_producer_version"] != version
                        or p.get("producer_revision") != version):
                    raise ContextualRecordError("stale producer release", record_id)
                value = torch.tensor(r["values"], dtype=torch.float32,
                                     device=self.order.weight.device)
            else:
                value, release_pin = self.release_corpus.values(
                    pin, span, version=version, scope=scope,
                    source_identity=self.source_identity,
                    payload=r.get("release_payload"))
                pin.update(release_pin)
                value = value.to(self.order.weight.device)
            if tuple(value.shape) != (SLOTS, WIDTH) or not torch.isfinite(value).all():
                raise ContextualRecordError("stored slot shape or value differs", record_id)
            pin["slots_sha256"] = tensor_digest(value)
            pins.append(pin)
            spans.append(span)
            values.append(value)
        batch = ValueBatch(torch.cat(values).detach(), tuple(spans), scope.scope_key, version,
                           pins=tuple(pins), store=store)
        self.validate(batch, scope)
        return batch

    def read(self, store, question, *, scope, wait_seconds=0.0):
        """Return a complete batch or a typed ticket; never expose partial values."""
        self._wait_bound(wait_seconds)
        if self.release_corpus is not None:
            if self.release_corpus.store is not store:
                raise ContextualMemoryError("producer corpus belongs to a different store")
            return self.release_corpus.read(self, question, scope=scope, wait_seconds=wait_seconds)
        self.cost["retrieval_calls"] += 1
        start = time.monotonic_ns()
        budget = ReadoutBudget(max_records=MAX_RECORDS, max_value_slots=self.spec.slots,
                               cold_wait_seconds=0.0, cache_bytes=0)
        result = read_scoped_memory(store, question, scope, budget)
        costs = {"retrieval_wall_ns": time.monotonic_ns() - start,
                 "read_file_bytes": result.metrics.total_io_bytes,
                 "read_file_calls": result.metrics.total_file_reads,
                 "cold_reserved_bytes": 0, "cold_reserved_files": 0,
                 "cold_completed_bytes": 0, "cold_completed_files": 0, "cold_wait_ns": 0}
        for key, value in costs.items():
            self.cost[key] += value
        if result.new_decision_boundary or result.truncated:
            self._evict_named([r.record_id for r in result.records if r.status != "ok"])
            raise ContextualReadFailure("incomplete or invalidated selection", costs)
        if not result.pending_cold:
            if (result.status not in ("ok", "degraded")
                    or any(r.status != "ok" for r in result.records)):
                self._evict_named([r.record_id for r in result.records if r.status != "ok"])
                raise ContextualReadFailure("incomplete contextual read", costs)
            try:
                batch = self._batch([_descriptor(r) for r in result.records], store, scope)
            except ContextualRecordError as error:
                if error.record_id:
                    self.feature_provider.evict_record(error.record_id)
                raise
            except Exception:
                self._evict_named([r.record_id for r in result.records])
                raise
            batch.read_costs = dict(costs)
            return batch
        pending_ids = {p.record_id for p in result.pending_cold}
        if any(r.status != "ok" and r.record_id not in pending_ids for r in result.records):
            self._evict_named([r.record_id for r in result.records
                               if r.status != "ok" and r.record_id not in pending_ids])
            raise ContextualReadFailure("partial selection refused", costs)
        token = uuid.uuid4().hex
        tickets = [asdict(store.ticket(p.ticket_id)) for p in result.pending_cold]
        doc = {"format": FORMAT + ".pending", "id": token,
               "store_root": str(store.root.resolve()), "scope": asdict(scope),
               "records": [_descriptor(r) for r in result.records], "tickets": tickets,
               "costs": costs, "status": "pending", "attempts": 0,
               "producer_release": self.encoder.release(), "max_attempts": 2}
        self._persist(store, token, doc)
        pending = self._pending(doc)
        return self.resume(store, pending, scope=scope, wait_seconds=wait_seconds) if wait_seconds else pending

    def begin_release_read(self, store, scope, records, payloads, pointer, *, wait_seconds):
        token = uuid.uuid4().hex
        costs = {key: 0 for key in (
            "cold_reserved_bytes", "cold_reserved_files", "cold_completed_bytes",
            "cold_completed_files", "cold_wait_ns", "read_file_bytes", "read_file_calls")}
        doc = {"format": FORMAT + ".pending", "id": token,
               "store_root": str(store.root.resolve()), "scope": asdict(scope),
               "records": records, "release_payloads": payloads, "release_pointer": pointer,
               "costs": costs, "status": "pending", "attempts": 0,
               "producer_release": self.encoder.release(), "max_attempts": 2}
        self._persist(store, token, doc)
        pending = self._pending(doc)
        return self.resume(store, pending, scope=scope, wait_seconds=wait_seconds) if wait_seconds else pending

    def _evict_named(self, record_ids):
        for record_id in {rid for rid in record_ids if rid}:
            self.feature_provider.evict_record(record_id)

    def _evict_failed_records(self, store, doc):
        """Evict only records whose own authority or staged payload is invalid.

        Healthy selected records keep their bounded feature cache. The worker's
        charged per-record payload failures are carried in
        ``failed_payload_records``; no secondary uncharged read is performed here.
        """
        invalid = set(doc.get("failed_payload_records") or ())
        for ticket in doc.get("tickets") or ():
            for pin in ticket.get("pinned") or ():
                record_id = pin.get("record_id")
                if not record_id:
                    continue
                row = store.connection.execute(
                    "SELECT r.*, c.current_generation FROM records r JOIN clusters c "
                    "ON r.cluster_id=c.cluster_id WHERE r.record_id=?",
                    (record_id,)).fetchone()
                if (row is None or row["status"] != "current"
                        or row["scope_key"] != pin.get("scope_key")
                        or int(row["revision"]) != int(pin.get("revision", -1))
                        or row["text_sha256"] != pin.get("text_sha256")):
                    invalid.add(record_id)
        if doc.get("release_payloads") or "release_pointer" in doc:
            # A failed/obsolete release batch is authority-stale as a whole.
            for record in doc.get("records") or ():
                record_id = record.get("record_id")
                if record_id:
                    invalid.add(record_id)
        evicted = []
        for record_id in sorted(invalid):
            self.feature_provider.evict_record(record_id)
            evicted.append(record_id)
        return evicted

    @staticmethod
    def _wait_bound(seconds):
        if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds) or not 0 <= seconds <= 5):
            raise ContextualMemoryError("cold wait must be bounded to 0..5 seconds")

    @staticmethod
    def _ticket_path(store, token):
        if len(token) != 32 or any(c not in "0123456789abcdef" for c in token):
            raise ContextualMemoryError("invalid contextual ticket identity")
        return store.root / "artifacts" / "contextual_v12" / "tickets" / (token + ".json")

    def _persist(self, store, token, doc):
        path = self._ticket_path(store, token)
        # Charge the descriptor write before it happens. Persisting the cumulative
        # reservation makes timeout/restart/failure cost visible to the caller.
        costs = doc["costs"]
        previous = costs.get("ticket_write_bytes_reserved", 0)
        costs["ticket_write_calls"] = costs.get("ticket_write_calls", 0) + 1
        size = 0
        for _ in range(10):
            costs["ticket_write_bytes_reserved"] = previous + size
            actual = len(json.dumps(doc, sort_keys=True, indent=2).encode()) + 1
            if actual == size:
                break
            size = actual
        # Ticket overlap and retained descriptor are charged to existing cold quota.
        store._charge("cold", "artifacts", size, "contextual_ticket:" + token + ":write",
                      "pending descriptor")
        start = time.monotonic_ns()
        try:
            atomic_json(path, doc)
        finally:
            store._release("contextual_ticket:" + token + ":write")
        store._set_component_locked("contextual_ticket:" + token, "cold", "artifacts", size,
                                    "pending descriptor")
        self.cost["durable_write_bytes"] += size
        self.cost["durable_write_ns"] += time.monotonic_ns() - start


    def _persist_charged_fallback(self, store, token, doc):
        """Persist a terminal receipt with quota, without repeating a failed reservation."""
        path = self._ticket_path(store, token)
        costs = doc["costs"]
        previous = costs.get("ticket_write_bytes_reserved", 0)
        costs["ticket_write_calls"] = costs.get("ticket_write_calls", 0) + 1
        size = len(json.dumps(doc, sort_keys=True, indent=2).encode()) + 1
        costs["ticket_write_bytes_reserved"] = previous + size
        component = "contextual_ticket:" + token + ":write"
        store._charge("cold", "artifacts", size, component, "pending descriptor")
        start = time.monotonic_ns()
        try:
            atomic_json(path, doc)
        finally:
            store._release(component)
        store._set_component_locked(component, "cold", "artifacts", size,
                                    "pending descriptor")
        self.cost["durable_write_bytes"] += size
        self.cost["durable_write_ns"] += time.monotonic_ns() - start
    @staticmethod
    def _pending(doc):
        return PendingContextualRead(doc["id"], doc["store_root"], doc["scope"], digest(doc),
                                     dict(doc["costs"]))

    def recover_pending(self, store, ticket_id, *, scope):
        """Recover the latest committed descriptor after an interrupted wait.

        This does not dispatch I/O or reset uncertain reservations. Original E
        authority is checked again after any resumed payload read.
        """
        doc, size = self._load_ticket(store, ticket_id)
        if (doc.get("format") != FORMAT + ".pending" or doc.get("id") != ticket_id
                or doc.get("store_root") != str(store.root.resolve())
                or doc.get("scope") != asdict(scope)
                or doc.get("producer_release") != self.encoder.release()
                or doc.get("status") != "pending"):
            raise ContextualReadFailure("pending recovery authority differs", doc.get("costs", {}))
        self._charge_ticket_read(doc, size)
        self._persist(store, ticket_id, doc)
        return self._pending(doc)

    def _load_ticket(self, store, ticket_id):
        path = self._ticket_path(store, ticket_id)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 256 * 1024:
            raise ContextualMemoryError("missing/substituted/oversized pending descriptor")
        payload = path.read_bytes()
        self.cost["ticket_descriptor_read_bytes"] += len(payload)
        self.cost["ticket_descriptor_read_calls"] += 1
        try:
            doc = json.loads(payload)
        except (UnicodeDecodeError, ValueError) as error:
            raise ContextualMemoryError("corrupt pending descriptor") from error
        if not isinstance(doc, dict) or not isinstance(doc.get("costs"), dict):
            raise ContextualMemoryError("invalid pending descriptor")
        return doc, len(payload)

    @staticmethod
    def _charge_ticket_read(doc, size):
        costs = doc["costs"]
        costs["ticket_descriptor_read_bytes"] = costs.get("ticket_descriptor_read_bytes", 0) + size
        costs["ticket_descriptor_read_calls"] = costs.get("ticket_descriptor_read_calls", 0) + 1

    def resume(self, store, pending, *, scope, wait_seconds=0.0):
        self._wait_bound(wait_seconds)
        if not isinstance(pending, PendingContextualRead):
            raise ContextualMemoryError("pending contextual ticket required")
        doc, size = self._load_ticket(store, pending.ticket_id)
        if (doc.get("format") != FORMAT + ".pending"
                or doc.get("store_root") != str(store.root.resolve())
                or pending.store_root != doc.get("store_root")
                or doc.get("scope") != asdict(scope) or pending.scope != doc.get("scope")
                or doc.get("producer_release") != self.encoder.release()
                or digest(doc) != pending.descriptor_sha256):
            raise ContextualReadFailure("pending scope/release/descriptor identity differs",
                                        doc.get("costs", {}))
        if doc.get("status") != "pending":
            raise ContextualReadFailure("ticket already terminal", doc["costs"])
        self._charge_ticket_read(doc, size)
        if wait_seconds == 0:
            self._persist(store, pending.ticket_id, doc)
            return self._pending(doc)
        if pending.ticket_id not in self._jobs:
            # Reserve the maximum I/O before dispatch. If the process dies, its
            # reservation remains charged as uncertain work; a retry adds a new one.
            release = doc.get("release_payloads")
            byte_count = (sum(p["bytes"] for p in release) if release
                          else sum(t["required_bytes"] for t in doc["tickets"]))
            if (doc["attempts"] >= doc["max_attempts"]
                    or any(t["required_bytes"] > t["max_bytes"] for t in doc.get("tickets", []))):
                raise ContextualReadFailure("cold attempt/byte budget exhausted", doc["costs"])
            files = (len(release) if release
                     else sum(1 + len(p["values"]) for t in doc["tickets"] for p in t["pinned"]))
            for key, amount in (("cold_reserved_bytes", byte_count), ("cold_reserved_files", files)):
                doc["costs"][key] += amount
                self.cost[key] += amount
            doc["attempts"] += 1
            self._persist(store, pending.ticket_id, doc)
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=1)
            fn, args = ((_release_payloads, release) if release
                        else (_cold_payloads, doc["tickets"]))
            self._jobs[pending.ticket_id] = self._pool.submit(fn, store.root, args)
        start = time.monotonic_ns()
        try:
            result = self._jobs[pending.ticket_id].result(timeout=wait_seconds)
            if len(result) == 4:
                staged, io, error, invalid_ids = result
            else:
                staged, io, error = result
                invalid_ids = set()
        except TimeoutError:
            elapsed = time.monotonic_ns() - start
            doc["costs"]["cold_wait_ns"] += elapsed
            self.cost["cold_wait_ns"] += elapsed
            self._persist(store, pending.ticket_id, doc)
            return self._pending(doc)
        elapsed = time.monotonic_ns() - start
        doc["costs"]["cold_wait_ns"] += elapsed
        self.cost["cold_wait_ns"] += elapsed
        self._jobs.pop(pending.ticket_id)
        for key, value in (("cold_completed_bytes", io["io_bytes"]),
                           ("cold_completed_files", io["file_reads"]),
                           ("read_file_bytes", io["io_bytes"]),
                           ("read_file_calls", io["file_reads"])):
            doc["costs"][key] += value
            self.cost[key] += value
        try:
            if error:
                if isinstance(error, dict):
                    raise next(iter(error.values()))
                raise ContextualMemoryError(error)
            if "release_payloads" in doc:
                if (self.release_corpus is None
                        or self.release_corpus._current()[0] != doc["release_pointer"]):
                    raise AuthorityRejection("stale_snapshot", "producer publication invalidated pending read")
                descriptors = [{**r, "release_payload": staged[r["record_id"]]}
                               for r in doc["records"]]
                batch = self._batch(descriptors, store, scope)
                doc["status"] = "complete"
                self._persist(store, pending.ticket_id, doc)
                batch.read_costs = dict(doc["costs"])
                return batch
            updates = {}
            for t in doc["tickets"]:
                ticket = RecallTicket(**{**t, "scope": MemoryScope(**t["scope"]),
                                          "outcome": Outcome(**t["outcome"])})
                status, reason, checks = store._validate_after_io(ticket, staged)
                if status != "ready":
                    raise AuthorityRejection(
                        "revoked_source" if "revoked" in reason or "superseded" in reason
                        else "stale_snapshot", reason)
                records = tuple(RecalledRecord(record_id=rid, text=staged[rid][0].decode(),
                    text_sha256=hashlib.sha256(staged[rid][0]).hexdigest(),
                    values=unpack_values(staged[rid][1]), value_slots=SLOTS,
                    value_sha256=tuple(hashlib.sha256(v).hexdigest() for v in staged[rid][1]),
                    placement="cold") for rid in ticket.record_ids)
                ready = RecallResult(ticket_id=ticket.ticket_id, selection_id=ticket.selection_id,
                    status="ready", outcome=Outcome("ok", "bounded_cold_ready"), records=records,
                    io_bytes=sum(len(staged[rid][0]) + sum(map(len, staged[rid][1]))
                                 for rid in ticket.record_ids),
                    checks=checks, new_decision_boundary=False,
                    file_reads=sum(1 + len(staged[rid][1]) for rid in ticket.record_ids))
                store._tickets[ticket.ticket_id] = {"ticket": replace(ticket, status="ready")}
                store._ticket_results[ticket.ticket_id] = ready
                result = resume_scoped_readout(store, ticket.ticket_id, scope=scope,
                    budget=ReadoutBudget(max_records=MAX_RECORDS, max_value_slots=self.spec.slots,
                                         cache_bytes=0))
                if (not result.ok or result.new_decision_boundary
                        or any(r.status != "ok" for r in result.records)):
                    raise AuthorityRejection("stale_snapshot", "post-I/O read invalidated")
                updates.update({r.record_id: _descriptor(r) for r in result.records})
            descriptors = [updates.get(r["record_id"], r) for r in doc["records"]]
            batch = self._batch(descriptors, store, scope)
        except Exception as error:
            doc["failed_payload_records"] = sorted(invalid_ids)
            try:
                self._evict_failed_records(store, doc)
            except Exception:
                # Best-effort selective eviction must not mask the typed failure.
                pass
            doc["status"] = "failed"
            doc["reason"] = str(error)
            try:
                self._persist(store, pending.ticket_id, doc)
            except Exception:
                # Preserve the terminal receipt even if charged persistence fails.
                    try:
                        self._persist_charged_fallback(store, pending.ticket_id, doc)
                    except Exception:
                        pass
            if isinstance(error, AuthorityRejection):
                raise AuthorityReadFailure(error.authority_reason, str(error), doc["costs"]) from error
            raise ContextualReadFailure(str(error), doc["costs"]) from error
        doc["status"] = "complete"
        self._persist(store, pending.ticket_id, doc)
        batch.read_costs = dict(doc["costs"])
        return batch

    # ------------------------------------------------------------------
    # Complete checkpoint and lifecycle

    def save(self, directory):
        self.feature_provider.validate_original()
        self.assert_optimizer_ownership()
        directory = Path(directory)
        directory.mkdir(exist_ok=False, parents=True)
        inner = super().save(directory / "components")
        manifest = {"format": FORMAT + ".checkpoint", "objective": self.objective_identity,
                    "execution_identity": self.execution_identity,
                    "slot_policy": SLOT_POLICY, "temporal_rule": "MemoryScope.cutoff",
                    "source_identity": self.source_identity,
                    "base": self.adapter.base_snapshot(),
                    "feature_contract": self.feature_provider.spec.identity(),
                    "adapter_contract": self.feature_provider.adapter_contract,
                    "tokenizer_contract": self.feature_provider.tokenizer_contract,
                    "layout": self.spec.identity(), "producer": PRODUCER,
                    "reader_key_mode": self.key_mode,
                    "producer_release": inner["producer_release"],
                    "representation_release": inner["representation_release"],
                    "producer_state_release": inner["producer_state_release"],
                    "contains_base_weights": False,
                    "original_weights_serialized": False,
                    "projection_cache": "none",
                    "feature_cache_serialized": False, "task_state_serialized": False,
                    "fitting_frozen": inner["fitting_frozen"],
                    "components_manifest_sha256": file_hash(directory / "components" / "manifest.json")}
        atomic_json(directory / "manifest.json", manifest)
        return {**manifest, "components": inner}

    def load(self, directory):
        directory = Path(directory)
        file_hash(directory / "manifest.json")
        manifest = json.loads((directory / "manifest.json").read_text())
        components_path = directory / "components" / "manifest.json"
        components_identity = file_hash(components_path)
        components = json.loads(components_path.read_text())
        expected = {"format": FORMAT + ".checkpoint", "objective": self.objective_identity,
                    "execution_identity": self.execution_identity,
                    "slot_policy": SLOT_POLICY, "temporal_rule": "MemoryScope.cutoff",
                    "source_identity": self.source_identity,
                    "base": self.adapter.base_snapshot(),
                    "feature_contract": self.feature_provider.spec.identity(),
                    "adapter_contract": self.feature_provider.adapter_contract,
                    "tokenizer_contract": self.feature_provider.tokenizer_contract,
                    "layout": self.spec.identity(), "producer": PRODUCER,
                    "reader_key_mode": self.key_mode,
                    "contains_base_weights": False,
                    "original_weights_serialized": False,
                    "projection_cache": "none",
                    "feature_cache_serialized": False, "task_state_serialized": False}
        if (any(manifest.get(k, "ordered" if k == "reader_key_mode" else None) != v
                for k, v in expected.items())
                or manifest.get("components_manifest_sha256") != components_identity):
            raise ContextualMemoryError("v12 objective/slots/checkpoint identity differs")
        if (type(manifest.get("fitting_frozen")) is not bool
                or manifest["fitting_frozen"] != components.get("fitting_frozen")
                or manifest.get("producer_release") != components.get("producer_release")
                or manifest.get("representation_release") != components.get("representation_release")
                or manifest.get("producer_state_release") != components.get("producer_state_release")):
            raise ContextualMemoryError("v12 outer checkpoint binding differs")
        result = super().load(directory / "components")
        self._release_cache = None
        self.feature_provider.clear()
        return {**manifest, "components": result}

    def close(self):
        """Release bounded caches and in-flight workers; no source value survives."""
        self.feature_provider.clear()
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
        self._jobs.clear()

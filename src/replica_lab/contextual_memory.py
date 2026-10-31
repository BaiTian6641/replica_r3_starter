"""Intrinsic memory v3 with contextual values v2 and separate durable state.

The byte transformer sees canonical source text only. Four ordered, width-256
slots summarize four contiguous source regions with full-span context. Retrieval
keys remain hash controls; they are never the values. A training graph lives for
one objective/update; durable writes, inference reads and decisions detach.
Memory owns its decoder readers; the external spine is optional and independent.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import time
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import nn

from .architecture.intrinsic_memory import IntrinsicMemorySpec, IntrinsicMemoryReader, IntrinsicMemoryContext
from .optimizers import NativeFusedAdamW
from .tiered_memory import EvidenceRecordDraft, MemoryScope, TieredMemoryStore, pack_values
from .tiered_readout import ReadoutBudget, read_scoped_memory

FORMAT = "replica.intrinsic_memory.implementation.v3"
LEGACY_FORMAT = "replica.contextual_memory.implementation.v2"
PRODUCER = "contextual_byte_transformer_v2"
REPRESENTATION = "four_ordered_contextual_256_v2"
CODEC = "contextual_fp32_slots_v2"
MAX_SPAN_BYTES = 256
MAX_CANONICAL_BYTES = 8192
SLOTS = 4
WIDTH = 256


class ContextualMemoryError(ValueError):
    pass

class AuthorityRejection(ContextualMemoryError):
    """Expected authority invalidation, never an operational read/write failure."""

    def __init__(self, authority_reason, message):
        super().__init__(message)
        self.authority_reason = authority_reason


def text_digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tensor_digest(tensor):
    # This is identity/serialization only; never used as a training value.
    return hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class CanonicalSpan:
    record_id: str
    source_id: str
    source_epoch: int
    source_order: int
    start: int
    end: int
    canonical_sha256: str
    text: str
    available_at: int
    valid_until: int | None = None
    supersedes: str | None = None

    def validate(self):
        if (not self.text or len(self.text.encode()) > MAX_SPAN_BYTES
                or self.end - self.start != len(self.text) or self.start < 0
                or self.source_order < 0 or self.source_epoch < 0 or self.available_at < 0
                or not self.record_id or not self.source_id or len(self.canonical_sha256) != 64
                or (self.valid_until is not None and self.valid_until <= self.available_at)):
            raise ContextualMemoryError("invalid bounded canonical span/provenance")
        return self


def canonical_spans(text: str, *, record_id: str, source_id: str, source_epoch: int,
                    available_at: int, valid_until: int | None = None) -> tuple[CanonicalSpan, ...]:
    if not text or len(text.encode()) > MAX_CANONICAL_BYTES:
        raise ContextualMemoryError("canonical text exceeds declared bound or is empty")
    spans = []
    start = 0
    while start < len(text):
        end = start
        size = 0
        while end < len(text) and size + len(text[end].encode()) <= MAX_SPAN_BYTES:
            size += len(text[end].encode())
            end += 1
        spans.append(CanonicalSpan(f"{record_id}.span{len(spans)}", source_id, source_epoch,
                     len(spans), start, end, text_digest(text), text[start:end],
                     available_at, valid_until).validate())
        start = end
    return tuple(spans)


class ContextualValueEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(257, WIDTH, padding_idx=0)
        self.position = nn.Embedding(MAX_SPAN_BYTES, WIDTH)
        layer = nn.TransformerEncoderLayer(WIDTH, 4, 512, dropout=0.0,
                    activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.output = nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, WIDTH, bias=False))
        self.slot_queries = nn.Parameter(torch.empty(SLOTS, WIDTH))
        nn.init.normal_(self.slot_queries, std=0.02)

    def forward(self, spans: Sequence[CanonicalSpan]) -> torch.Tensor:
        if not spans:
            raise ContextualMemoryError("encoder requires admitted source spans")
        device = self.embedding.weight.device
        values = []
        for span in spans:
            span.validate()
            ids = torch.tensor([b + 1 for b in span.text.encode()], device=device)
            hidden = self.embedding(ids) + self.position(torch.arange(len(ids), device=device))
            hidden = self.transformer(hidden.unsqueeze(0))[0]
            slots = []
            for slot in range(SLOTS):
                # Nonempty windows even for one-byte records; source order is
                # explicit, while the contextual transform sees the full span.
                begin = min(len(ids) - 1, slot * len(ids) // SLOTS)
                end = max(begin + 1, (slot + 1) * len(ids) // SLOTS)
                region = hidden[begin:end]
                weights = torch.softmax(region @ self.slot_queries[slot] / WIDTH**0.5, dim=0)
                slots.append(self.output((weights[:, None] * region).sum(0)))
            values.append(torch.stack(slots))
        return torch.stack(values)

    def release(self) -> str:
        digest = hashlib.sha256()
        for name, tensor in self.state_dict().items():
            digest.update(name.encode())
            digest.update(tensor_digest(tensor).encode())
        return "v2-" + digest.hexdigest()


@dataclass
class ValueBatch:
    slots: torch.Tensor
    spans: tuple[CanonicalSpan, ...]
    scope_key: str
    producer_version: str
    representation: str = REPRESENTATION
    codec: str = CODEC
    # Durable read pins are rechecked at each injection boundary.
    pins: tuple[Mapping, ...] = ()
    store: TieredMemoryStore | None = None
    training: bool = False


class ContextualMemoryPath:
    """Exactly one optimizer owner for producer, ordered reader and memory ports.

    All unrelated extension parameters and all original parameters are frozen.
    P has its own owner and is absent from this readout stage. After qualifying
    readout, freeze_readout removes every phi tensor from the training graph.
    """
    def __init__(self, adapter, *, lr=1e-3):
        self.spec = IntrinsicMemorySpec(hidden_size=adapter.config.geometry.hidden_size)
        self.adapter = adapter
        for parameter in adapter.base.parameters():
            parameter.requires_grad_(False)
        for parameter in adapter.extension.parameters():
            parameter.requires_grad_(False)
        device = next(adapter.extension.parameters()).device
        self.reader = IntrinsicMemoryReader.install(adapter, self.spec)
        self.encoder = ContextualValueEncoder().to(device)
        self.order = nn.Embedding(self.spec.slots, WIDTH).to(device)
        nn.init.normal_(self.order.weight, std=0.02)
        for parameter in self.reader.ports.parameters():
            parameter.requires_grad_(True)
        groups = [{"params": list(self.encoder.parameters()), "lr": lr, "name": "producer"},
                  {"params": list(self.order.parameters()) + list(self.reader.ports.parameters()),
                   "lr": lr, "name": "reader"}]
        for group in groups:
            group.update(algorithm="adamw_fused", betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
        names = {id(p): "producer."+n for n,p in self.encoder.named_parameters()}
        names.update({id(p): "order."+n for n,p in self.order.named_parameters()})
        names.update({id(p): "reader."+n for n,p in self.reader.ports.named_parameters()})
        for group in groups:
            group.update(group_name=group["name"], scope="contextual_phi",
                         param_names=[names[id(p)] for p in group["params"]])
        self.optimizer = NativeFusedAdamW(groups)
        self.cost = {"producer_bytes": 0, "producer_calls": 0, "value_slots_read": 0,
                     "producer_wall_ns": 0, "retrieval_wall_ns": 0,
                     "fallback_bytes": 0, "fallback_tokens": 0, "fallback_calls": 0,
                     "read_file_bytes": 0, "read_file_calls": 0, "revalidations": 0}
        owned = [id(p) for group in self.optimizer.param_groups for p in group["params"]]
        trainable = [id(p) for p in self.encoder.parameters()] + [id(p) for p in self.order.parameters()]
        trainable += [id(p) for p in adapter.extension.parameters() if p.requires_grad]
        if len(owned) != len(set(owned)) or set(owned) != set(trainable):
            raise ContextualMemoryError("trainable parameters do not have exactly one optimizer owner")

    def training_values(self, spans, *, scope: MemoryScope) -> ValueBatch:
        for s in spans:
            s.validate()
            if s.available_at > scope.as_of or (s.valid_until is not None and scope.as_of >= s.valid_until):
                raise ContextualMemoryError("future/expired training span")
        # No NumPy, hashing, disk or .detach on the value path.
        slots = self._encode(spans).reshape(-1, WIDTH)
        return ValueBatch(slots, tuple(spans), scope.scope_key, "training-current", training=True)

    def _encode(self, spans):
        self.cost["producer_calls"] += 1
        self.cost["producer_bytes"] += sum(len(s.text.encode()) for s in spans)
        start = time.monotonic_ns()
        values = self.encoder(spans)
        # Dispatch wall time; asynchronous device work is covered by the native
        # end-to-end profiler, not misreported here as kernel execution time.
        self.cost["producer_wall_ns"] += time.monotonic_ns() - start
        return values

    def training_from_stored(self, batch, *, scope):
        self.validate(batch, scope)
        fresh = self.training_values(batch.spans, scope=scope)
        fresh.pins, fresh.store = batch.pins, batch.store
        return fresh

    def provenance(self, span, version):
        metadata = {k:v for k,v in asdict(span).items() if k not in ("text", "source_id")}
        return {**metadata, "source": span.source_id, "text_sha256": text_digest(span.text),
                "producer_revision": version, "representation_version": REPRESENTATION,
                "codec_version": CODEC, "payload_type": "observation",
                "slot_spans": [[i, span.start, span.end, text_digest(span.text)] for i in range(SLOTS)]}

    def admit(self, store, span, *, scope, cluster="contextual"):
        span.validate()
        version = self.encoder.release()
        with torch.no_grad():
            values = self._encode([span])[0].detach().cpu().tolist()
        provenance = self.provenance(span, version)
        draft = EvidenceRecordDraft(span.record_id, span.text, span.source_id, span.source_epoch,
            span.available_at, valid_to=span.valid_until, authenticity="source_report",
            verification_state="verified_source", values=tuple(tuple(x) for x in values),
            value_producer_id=PRODUCER, value_producer_version=version,
            source_lineage=(json.dumps(provenance, sort_keys=True, separators=(",", ":")),))
        return store.admit_cluster(cluster, scope=scope, records=[draft])

    def read(self, store, question, *, scope) -> ValueBatch:
        start = time.monotonic_ns()
        result = read_scoped_memory(store, question, scope, ReadoutBudget(
            max_records=self.spec.slots // SLOTS,
            max_value_slots=self.spec.slots))
        self.cost["retrieval_wall_ns"] += time.monotonic_ns() - start
        self.cost["read_file_bytes"] += result.metrics.total_io_bytes
        self.cost["read_file_calls"] += result.metrics.total_file_reads
        if result.new_decision_boundary or result.pending_cold or result.status not in ("ok", "degraded"):
            raise ContextualMemoryError("incomplete contextual read")
        spans, values, pins = [], [], []
        version = self.encoder.release()
        records = sorted(result.records, key=lambda r: (
            r.source_id, r.source_epoch, r.source_provenance.get("source_order", 0)))
        for r in records:
            p = r.source_provenance
            if (r.status != "ok" or r.values is None or r.value_producer_id != PRODUCER
                    or r.value_producer_version != version or p.get("producer_revision") != version
                    or p.get("representation_version") != REPRESENTATION or p.get("codec_version") != CODEC
                    or r.scope_key != scope.scope_key or p.get("record_id") != r.record_id
                    or p.get("text_sha256") != r.text_sha256 or p.get("supersedes") != r.supersedes):
                raise ContextualMemoryError("stale producer/codec, correction or value authority")
            span_metadata = {**p, "text": r.text, "source_id": r.source_id}
            span = CanonicalSpan(**{k: span_metadata[k] for k in CanonicalSpan.__dataclass_fields__}).validate()
            if self.provenance(span, version)["slot_spans"] != p.get("slot_spans"):
                raise ContextualMemoryError("slot source order/provenance differs")
            slots = torch.tensor(r.values, dtype=torch.float32, device=self.order.weight.device)
            if tuple(slots.shape) != (SLOTS, WIDTH):
                raise ContextualMemoryError("stored contextual slot shape differs")
            pins.append({"record_id": r.record_id, "cluster_id": r.cluster_id,
                         "generation": r.cluster_generation, "revision": r.revision,
                         "text_sha256": r.text_sha256, "slots_sha256": tensor_digest(slots)})
            spans.append(span)
            values.append(slots)
        if not values:
            raise ContextualMemoryError("no eligible contextual values")
        batch = ValueBatch(torch.cat(values).detach(), tuple(spans), scope.scope_key, version,
                           pins=tuple(pins), store=store)
        self.validate(batch, scope)
        return batch

    def validate(self, batch, scope):
        if (batch.scope_key != scope.scope_key or batch.representation != REPRESENTATION
                or batch.codec != CODEC or batch.slots.ndim != 2
                or batch.slots.shape != (len(batch.spans) * SLOTS, WIDTH)
                or len(batch.slots) > self.spec.slots
                or not torch.isfinite(batch.slots).all()):
            raise ContextualMemoryError("scope, representation or slot budget differs")
        if not batch.training and batch.producer_version != self.encoder.release():
            raise ContextualMemoryError("stale producer release")
        for i, span in enumerate(batch.spans):
            if span.available_at > scope.as_of or (span.valid_until is not None and scope.as_of >= span.valid_until):
                raise ContextualMemoryError("future or expired memory")
            if batch.store is not None:
                pin = batch.pins[i]
                row = batch.store.connection.execute(
                    "SELECT r.*, c.current_generation FROM records r JOIN clusters c ON r.cluster_id=c.cluster_id WHERE record_id=?",
                    (pin["record_id"],)).fetchone()
                self.cost["revalidations"] += 1
                if (row is None or row["status"] != "current" or row["scope_key"] != scope.scope_key
                        or row["current_generation"] != pin["generation"] or row["revision"] != pin["revision"]
                        or row["text_sha256"] != pin["text_sha256"]
                        or (not batch.training and tensor_digest(batch.slots[i*SLOTS:(i+1)*SLOTS]) != pin["slots_sha256"])):
                    raise ContextualMemoryError("correction/revocation/generation invalidated memory")

    def forward(self, ids, mask, batch: ValueBatch | None, *, scope, enabled=True, p_slots=None):
        if not enabled or (batch is None and p_slots is None):
            return self.adapter.forward_text(ids, mask, use_projector=False, use_ports=False, use_spine=False)
        if batch is not None:
            self.validate(batch, scope)
        length = ids.shape[1]
        slots = (batch.slots + self.order(torch.arange(len(batch.slots), device=ids.device))
                 if batch is not None else None)
        if p_slots is not None:
            if p_slots.shape != (1, WIDTH) or not torch.isfinite(p_slots).all():
                raise ContextualMemoryError("P slots must be one finite width-256 value")
            slots = p_slots if slots is None else torch.cat((slots, p_slots))
        if len(slots) > self.spec.slots:
            raise ContextualMemoryError("combined E/P slot budget exceeded")
        slot_count = len(slots)
        slots = slots[None, None].expand(ids.shape[0], length, -1, -1)
        available = torch.ones(slots.shape[:3], device=ids.device, dtype=torch.bool)
        context = IntrinsicMemoryContext(self.reader, slots, available,
            validate=lambda: self.validate(batch, scope) if batch is not None else None)
        self.cost["value_slots_read"] += slot_count
        return self.adapter.forward_text(ids, mask, use_projector=False, use_ports=False,
                                        use_spine=False, intrinsic_memory=context)

    def exact_fallback(self, batch, *, scope, tokenizer, span_indices=None):
        self.validate(batch, scope)
        indices = list(range(len(batch.spans))) if span_indices is None else list(span_indices)
        if any(i < 0 or i >= len(batch.spans) for i in indices) or len(indices) != len(set(indices)):
            raise ContextualMemoryError("invalid exact-span fallback selection")
        text = "\n".join(batch.spans[i].text for i in indices)
        self.cost["fallback_calls"] += 1
        self.cost["fallback_bytes"] += len(text.encode())
        self.cost["fallback_tokens"] += len(tokenizer(text))
        return text

    def freeze_readout(self):
        for module in (self.encoder, self.order, self.reader.ports):
            for p in module.parameters():
                p.requires_grad_(False)
                p.grad = None
        self.optimizer = None

    def correct(self, store, old_record_id, span, *, scope):
        version = self.encoder.release()
        if span.supersedes != old_record_id:
            raise ContextualMemoryError("correction predecessor differs")
        with torch.no_grad():
            slots = self._encode([span])[0].detach().cpu().tolist()
        return store.correct_record(old_record_id, correction_id=span.record_id, text=span.text,
            source_id=span.source_id, source_epoch=span.source_epoch, observed_at=span.available_at,
            valid_to=span.valid_until, scope=scope, values=slots,
            value_producer_id=PRODUCER, value_producer_version=version,
            source_provenance=self.provenance(span, version))

    def save(self, directory):
        from safetensors.torch import save_file
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=False)
        tensors = {"producer."+k: v.detach().cpu().contiguous() for k,v in self.encoder.state_dict().items()}
        tensors.update({"order."+k: v.detach().cpu().contiguous() for k,v in self.order.state_dict().items()})
        tensors.update({"intrinsic_memory.reader."+k: v.detach().cpu().contiguous() for k,v in self.reader.ports.state_dict().items()})
        save_file(tensors, str(path / "extension.safetensors"))
        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), path / "optimizer.pt")
        manifest = {"format": FORMAT, "base": self.adapter.base_snapshot(),
                    "producer_version": self.encoder.release(), "representation": REPRESENTATION,
                    "codec": CODEC, "sites": [3,7], "contains_base_weights": False,
                    "owner": "intrinsic_memory", "external_spine_required": False,
                    "weights_sha256": hashlib.sha256((path / "extension.safetensors").read_bytes()).hexdigest()}
        manifest["optimizer_sha256"] = (hashlib.sha256((path / "optimizer.pt").read_bytes()).hexdigest()
                                         if self.optimizer is not None else None)
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
        return manifest

    def load(self, directory, *, migrate_legacy=False):
        from safetensors.torch import load_file
        path = Path(directory)
        m = json.loads((path / "manifest.json").read_text())
        legacy = m.get("format") == LEGACY_FORMAT
        if legacy and not migrate_legacy:
            raise ContextualMemoryError("legacy spine-owned readout requires explicit migration")
        if (m.get("format") not in (FORMAT, LEGACY_FORMAT) or m.get("base") != self.adapter.base_snapshot()
                or (not legacy and (m.get("owner") != "intrinsic_memory" or m.get("external_spine_required") is not False))
                or m.get("representation") != REPRESENTATION or m.get("codec") != CODEC
                or m.get("sites") != [3,7] or m.get("contains_base_weights") is not False
                or hashlib.sha256((path / "extension.safetensors").read_bytes()).hexdigest() != m["weights_sha256"]):
            raise ContextualMemoryError("extension-only reload binding differs")
        tensors = load_file(str(path / "extension.safetensors"))
        if legacy:
            tensors = {("intrinsic_memory."+k if k.startswith("reader.") else k):v
                       for k,v in tensors.items()}
        modules = {"producer": self.encoder, "order": self.order,
                   "intrinsic_memory.reader": self.reader.ports}
        if set(tensors) != {prefix+"."+k for prefix,module in modules.items() for k in module.state_dict()}:
            raise ContextualMemoryError("extension-only tensor inventory differs")
        for prefix,module in modules.items():
            module.load_state_dict({k[len(prefix)+1:]:v for k,v in tensors.items() if k.startswith(prefix+".")}, strict=True)
        if self.encoder.release() != m["producer_version"]:
            raise ContextualMemoryError("producer release mismatch after reload")
        if m.get("optimizer_sha256"):
            if self.optimizer is None or hashlib.sha256((path / "optimizer.pt").read_bytes()).hexdigest() != m["optimizer_sha256"]:
                raise ContextualMemoryError("optimizer continuation binding differs")
            self.optimizer.load_state_dict(torch.load(path / "optimizer.pt", weights_only=True))
        return m

    def migrate_legacy(self, source, destination):
        """Explicitly preserve v2 weights/moments while changing only ownership.

        The source artifact is never edited. This is not a new qualification.
        """
        old = self.load(source, migrate_legacy=True)
        if old["format"] != LEGACY_FORMAT:
            raise ContextualMemoryError("migration requires the legacy v2 format")
        new = self.save(destination)
        receipt = {"format": "replica.intrinsic_memory.migration.v1",
                   "from_format": LEGACY_FORMAT, "to_format": FORMAT,
                   "source_manifest_sha256": hashlib.sha256((Path(source)/"manifest.json").read_bytes()).hexdigest(),
                   "destination_manifest_sha256": hashlib.sha256((Path(destination)/"manifest.json").read_bytes()).hexdigest(),
                   "parameter_values_changed": False, "optimizer_state_preserved": old.get("optimizer_sha256") is not None,
                   "native_qualification": False}
        (Path(destination)/"migration.json").write_text(json.dumps(receipt,indent=2)+"\n")
        return receipt

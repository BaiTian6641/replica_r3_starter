"""New producer/representation identity without changing sealed v2–v11 state.

The trainable value path is live torch tensors from the frozen-feature boundary
to the response loss. Only declared durable/inference boundaries detach. A
reader refuses a stale producer release, a changed source/adapter/tokenizer
feature contract, an unavailable or corrected record and a mixed generation.
"""
from dataclasses import dataclass
import json

import torch

from .contextual_memory import (
    CanonicalSpan, ContextualMemoryError, ContextualMemoryPath, SLOTS, WIDTH,
    tensor_digest, text_digest)
from .contextual_source_features_v12 import (
    CODEC, PRODUCER, REPRESENTATION, SourceFeatureValueEncoder)
from .contextual_memory import ValueBatch as PreviousValueBatch

VALUE_PROVENANCE_FORMAT = "replica.contextual_value_provenance.v12"
MAX_LINEAGE_BYTES = 1024


def lineage_json(provenance):
    """Serialize the current provenance under the store's bounded lineage rule."""
    payload = json.dumps(provenance, sort_keys=True)
    if len(payload) > MAX_LINEAGE_BYTES:
        raise ContextualMemoryError("value provenance exceeds the durable lineage bound")
    return payload

__all__ = ["CODEC", "PRODUCER", "REPRESENTATION", "FeatureIdentityPath", "ValueBatch",
           "SourceFeatureValueEncoder", "SLOTS", "WIDTH", "CanonicalSpan",
           "lineage_json", "MAX_LINEAGE_BYTES", "VALUE_PROVENANCE_FORMAT"]


@dataclass
class ValueBatch(PreviousValueBatch):
    representation: str = REPRESENTATION
    codec: str = CODEC


class FeatureIdentityPath(ContextualMemoryPath):
    """Canonical-source contextual values with fresh v12 identities."""

    def training_values(self, spans, *, scope):
        spans = tuple(spans)
        if not spans:
            raise ContextualMemoryError("bounded nonempty source selection required")
        if len(spans) * SLOTS > self.spec.slots:
            raise ContextualMemoryError("two-span/eight-slot source budget exceeded")
        if len({span.record_id for span in spans}) != len(spans):
            raise ContextualMemoryError("duplicate source record in one selection")
        for span in spans:
            span.validate()
            if span.available_at > scope.cutoff or (span.valid_until is not None
                                                    and scope.cutoff >= span.valid_until):
                raise ContextualMemoryError("future/expired training span")
        slots = self._encode(spans).reshape(-1, WIDTH)
        batch = ValueBatch(slots, spans, scope.scope_key, "training-current", training=True)
        self.validate(batch, scope)
        return batch

    def provenance(self, span, version):
        return {
            "record_id": span.record_id, "source": span.source_id,
            "source_epoch": span.source_epoch, "source_order": span.source_order,
            "start": span.start, "end": span.end,
            "canonical_sha256": span.canonical_sha256,
            "text_sha256": text_digest(span.text),
            "available_at": span.available_at, "valid_until": span.valid_until,
            "supersedes": span.supersedes,
            "producer_revision": version,
            "representation_version": REPRESENTATION,
            "codec_version": CODEC,
            **self.encoder.provider.provenance(span),
        }

    def validate(self, batch, scope):
        if not isinstance(batch, ValueBatch):
            raise ContextualMemoryError("complete ValueBatch required")
        if (not batch.scope_key or batch.scope_key != scope.scope_key
                or batch.representation != REPRESENTATION or batch.codec != CODEC
                or not batch.spans or batch.slots.ndim != 2 or batch.slots.dtype != torch.float32
                or batch.slots.shape != (len(batch.spans) * SLOTS, WIDTH)
                or len(batch.spans) * SLOTS > self.spec.slots
                or not torch.isfinite(batch.slots).all()):
            raise ContextualMemoryError("scope, v12 representation, pins or slot budget differs")
        if not batch.training and batch.slots.requires_grad:
            raise ContextualMemoryError("durable/inference values must be detached")
        if batch.store is not None and len(batch.pins) != len(batch.spans):
            raise ContextualMemoryError("complete durable authority pins required")
        self.encoder.provider.validate_original()
        if not batch.training and batch.producer_version != self.encoder.release():
            raise ContextualMemoryError("stale producer release")
        for index, span in enumerate(batch.spans):
            span.validate()
            if (span.available_at > scope.cutoff
                    or (span.valid_until is not None and scope.cutoff >= span.valid_until)):
                self.encoder.provider.evict_record(span.record_id)
                raise ContextualMemoryError("future or expired memory")
            if batch.store is None:
                continue
            pin = batch.pins[index]
            if not isinstance(pin, dict) or pin.get("record_id") != span.record_id:
                raise ContextualMemoryError("durable authority pin differs from its span")
            row = batch.store.connection.execute(
                "SELECT r.*, c.current_generation FROM records r JOIN clusters c "
                "ON r.cluster_id=c.cluster_id WHERE record_id=?",
                (pin["record_id"],)).fetchone()
            self.cost["revalidations"] += 1
            if (row is None or row["status"] != "current" or row["scope_key"] != scope.scope_key
                    or row["current_generation"] != pin["generation"]
                    or row["revision"] != pin["revision"]
                    or row["text_sha256"] != pin["text_sha256"]
                    or (not batch.training
                        and tensor_digest(batch.slots[index*SLOTS:(index+1)*SLOTS])
                            != pin["slots_sha256"])):
                self.encoder.provider.evict_record(span.record_id)
                raise ContextualMemoryError("correction/revocation/generation invalidated memory")
        return True

    def correct(self, store, old_record_id, span, *, scope):
        span.validate()
        if (span.supersedes != old_record_id or span.available_at > scope.cutoff
                or (span.valid_until is not None and scope.cutoff >= span.valid_until)):
            raise ContextualMemoryError("correction predecessor or time differs")
        version = self.encoder.release()
        # Correcting a source invalidates its feature and tokenizer caches even
        # when the old payload remains on disk.
        self.encoder.provider.evict_record(old_record_id)
        with torch.no_grad():
            slots = self._encode([span])[0].detach().cpu().tolist()
        result = store.correct_record(old_record_id, correction_id=span.record_id, text=span.text,
            source_id=span.source_id, source_epoch=span.source_epoch, observed_at=span.available_at,
            valid_to=span.valid_until, scope=scope, values=slots,
            value_producer_id=PRODUCER, value_producer_version=version,
            source_provenance=json.loads(lineage_json(self.provenance(span, version))))
        self.encoder.provider.evict_record(span.record_id)
        return result

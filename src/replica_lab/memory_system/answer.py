"""Host answer-path facade for exact-source and opt-in pointer memory answers.

The pointer path is never a default. If pointer copy is disabled or unavailable,
the facade returns exact source spans for citation/code use, not a fabricated
answer. The model never authorizes success and the host never picks a value on
the model's behalf.
"""
from __future__ import annotations

from dataclasses import dataclass

from .session import PointerAnswer


@dataclass(frozen=True)
class MemoryAnswer:
    text: str
    method: str  # pointer_copy | exact_source | unavailable
    record_ids: tuple[str, ...]
    span_hashes: tuple[str, ...]
    source_ids: tuple[str, ...]
    producer: str | None
    exact_source: bool


class MemoryAnswerService:
    """One trusted host's explicit memory-answer surface."""

    def __init__(self, session, *, pointer_enabled: bool = False,
                 exact_source_fallback: bool = True):
        if not isinstance(pointer_enabled, bool) or not isinstance(exact_source_fallback, bool):
            raise ValueError('answer switches must be boolean')
        self.session = session
        self.pointer_enabled = pointer_enabled
        self.exact_source_fallback = exact_source_fallback

    def answer(self, request) -> MemoryAnswer:
        self.session.verify(request)
        frame = request.frame
        if frame is None:
            raise ValueError('answer requires an owned request frame')
        if self.pointer_enabled and frame.copy is not None and frame.branch_enabled('copy'):
            pointer: PointerAnswer = self.session.pointer_answer(request)
            return MemoryAnswer(
                text=pointer.text, method='pointer_copy',
                record_ids=(pointer.record_id,),
                span_hashes=(pointer.span_hash,),
                source_ids=pointer.source_ids, producer=pointer.producer,
                exact_source=True)
        if not self.exact_source_fallback:
            return MemoryAnswer('', 'unavailable', (), (), (), None, False)
        spans = tuple(frame.memory.spans)
        if not spans:
            return MemoryAnswer('', 'unavailable', (), (), (), None, False)
        texts = []
        record_ids, span_hashes, source_ids = [], [], []
        for span in spans:
            exact = self.session.exact_text(request, span)
            texts.append(exact.text)
            record_ids.append(exact.record_id)
            span_hashes.append(exact.span_hash)
            source_ids.append(exact.source_hash)
        return MemoryAnswer(
            text='\n'.join(texts), method='exact_source',
            record_ids=tuple(dict.fromkeys(record_ids)),
            span_hashes=tuple(span_hashes),
            source_ids=tuple(dict.fromkeys(source_ids)),
            producer=frame.memory.producer, exact_source=True)

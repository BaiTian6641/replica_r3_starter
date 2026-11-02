"""Real generation/source integration on independent random CPU fixtures.

No gate qualification, native capability, or answer-quality claim. The tiny
tokenizer is deterministic and evidence-sensitive, not a language tokenizer.
"""
from copy import deepcopy
import hashlib
import json

from .contextual_v12_fixture import canonical_span, open_v12_store, scope
from .intrinsic_memory_acceptance_v13 import render_chronological
from .intrinsic_memory_acceptance_v15 import chronological_correction, expect_invalidation
from .intrinsic_memory_correction_v18 import (
    correction_evaluation_view, fresh_revocation_check, publish_correction)
from .intrinsic_memory_coverage_v12 import COMPOUND_FAMILIES, _question_compound, _render_compound
from .intrinsic_memory_cpu_fixture_v18 import make_engine
from .intrinsic_memory_diagnostic_v12 import DiagnosticEngineV12, prompt
from .intrinsic_memory_native_v12 import run_correction_probe
from .intrinsic_memory_teacher_v12 import TeacherEngineV12


FAMILIES = ("direct", *COMPOUND_FAMILIES)


class GenerationBackend:
    def __init__(self):
        self.prompts = []

    def encode(self, text):
        self.prompts.append(text)
        return [3 + byte % 61 for byte in hashlib.sha256(text.encode()).digest()[:16]]

    def answer_ids(self, text):
        return [3 + byte % 61 for byte in hashlib.sha256(text.encode()).digest()[:16]]

    def decode(self, ids):
        return " ".join(f"token{token}" for token in ids)


def generation_engine(output):
    engine = make_engine(output)
    engine.backend = GenerationBackend()
    engine.max_new_tokens = 2
    # Exercise the exact method inherited by the native teacher controller.
    assert engine.generate.__func__ is TeacherEngineV12.generate is DiagnosticEngineV12.generate
    return engine


def correction_row(family, unknown):
    # Keep fixture IDs inside the existing bounded durable-provenance envelope.
    row = {"world_id": f"cg{FAMILIES.index(family)}{int(unknown)}", "query_entity": "A",
           "question": "What is the recorded color of A?", "rule_family": family,
           "answer": "unknown" if unknown else "red", "chronology": 1}
    if family == "direct":
        row["facts"] = [{"entity": "B" if unknown else "A", "color": "red", "revision": 1}]
        row["text"] = render_chronological(row["facts"])
    else:
        row["facts"] = {"anchor": "A", "target": "B", "answer": "red",
                        "anchor_colour": "blue" if family == COMPOUND_FAMILIES[0] else "red",
                        "unrecorded_target": unknown}
        row["text"] = _render_compound(row["facts"], family)
        row["question"] = _question_compound(row["facts"], family)
    return row


def admit_original(engine, store, row, sc):
    spans = canonical_span(row["text"], record_id=row["world_id"], source_id=row["world_id"])
    assert len(spans) == 1
    assert engine.path.admit(store, spans[0], scope=sc).outcome.ok
    return engine.read(store, sc, row["question"])


def controller_case(output, family, unknown):
    """Call the same correction stage as run_main, without mocking generation."""
    engine = generation_engine(output)
    store = open_v12_store(engine.output / "store")
    sc = scope("generation")
    row = correction_row(family, unknown)
    original = deepcopy(row)
    try:
        old = admit_original(engine, store, row, sc)
        before = dict(engine.ledger.snapshot()["reserved"])
        receipt = run_correction_probe(engine, row, "green", store, sc, old)
        assert row == original
        view = receipt["evaluation_view"]
        publication = receipt["correction_publication"]
        assert view["text"] == publication["current_text"]
        assert engine.backend.prompts == [prompt(view, exact=False), prompt(view, exact=True),
            prompt({**row, "text": "", "answer": "unknown"}, exact=False)]
        history_bytes = (engine.output / publication["history_path"]).read_bytes()
        assert hashlib.sha256(history_bytes).hexdigest() == publication["history_sha256"]
        assert json.loads(history_bytes)["row"] == chronological_correction(row, "green")
        assert json.loads(history_bytes)["row"]["text"] != view["text"]
        assert receipt["stale_snapshot_rejected"] and receipt["revoked_retrieval_rejected"]
        assert receipt["fresh_revocation_retrieval"]["selected_record_ids"] == []
        answers = [receipt[key] for key in ("corrected_contextual", "corrected_exact", "after_revocation")]
        assert all(answer["generated_tokens"] > 0 for answer in answers)
        after = engine.ledger.snapshot()["reserved"]
        forwards = after["forwards"] - before["forwards"]
        source_forwards = after["source_feature_forwards"] - before["source_feature_forwards"]
        assert source_forwards == 1  # New canonical correction source; never confuse with decoding.
        assert forwards - source_forwards == sum(answer["generated_tokens"] for answer in answers)
        return {"family": family, "unknown_origin": unknown,
                "generation_forwards": forwards - source_forwards, "source_forwards": source_forwards,
                "receipt": receipt, "native_capability_claim": False}
    finally:
        store.close()
        engine.path.close()


def repeated_case(output, family, unknown):
    engine = generation_engine(output)
    store = open_v12_store(engine.output / "store")
    sc = scope("repeated_generation")
    row = correction_row(family, unknown)
    retained = {}
    try:
        old = admit_original(engine, store, row, sc)
        for color in ("green", "gold", "blue"):
            prior = deepcopy(row)
            corrected = chronological_correction(row, color)
            frozen = deepcopy(corrected)
            span, publication = publish_correction(engine, store, old, corrected, sc)
            fresh = engine.read(store, sc, row["question"])
            view = correction_evaluation_view(corrected, span, publication)
            assert view["text"] == "".join(item.text for item in fresh.spans)
            assert view["provenance"]["predecessor"] == old.spans[0].record_id
            for control in ("contextual", "exact_text"):
                # Reproduce the original defect: historical text is still refused.
                try:
                    engine.generate(corrected, fresh, sc, control=control)
                except ValueError as exc:
                    assert "selected canonical spans" in str(exc)
                else:
                    raise AssertionError("historical text bypassed canonical-span validation")
                expect_invalidation("stale_snapshot", lambda: engine.generate(view, old, sc, control=control))
                answer = engine.generate(view, fresh, sc, control=control)
                assert answer["first_logits"] is not None and answer["generated_tokens"] > 0
            assert engine.backend.prompts[-2:] == [prompt(view, exact=False), prompt(view, exact=True)]
            history_path = engine.output / publication["history_path"]
            payload = history_path.read_bytes()
            assert hashlib.sha256(payload).hexdigest() == publication["history_sha256"]
            assert json.loads(payload)["row"] == corrected
            assert row == prior and corrected == frozen
            retained[history_path] = payload
            assert all(path.read_bytes() == contents for path, contents in retained.items())
            row, old = corrected, fresh
        store.revoke_record(span.record_id, reason="synthetic", scope=sc)
        for control in ("contextual", "exact_text"):
            expect_invalidation("stale_snapshot", lambda: engine.generate(view, fresh, sc, control=control))
        assert fresh_revocation_check(store, row["question"], sc, [span.record_id])["selected_record_ids"] == []
        assert all(path.read_bytes() == contents for path, contents in retained.items())
        return {"family": family, "unknown_origin": unknown, "revisions": len(retained),
                "history_sha256": [hashlib.sha256(value).hexdigest() for value in retained.values()],
                "native_capability_claim": False}
    finally:
        store.close()
        engine.path.close()

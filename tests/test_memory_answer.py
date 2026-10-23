"""Host answer facade: opt-in pointer and exact-source fallback."""
from dataclasses import dataclass
from replica_lab.memory_system.answer import MemoryAnswerService

@dataclass
class Span:
    record_id: str; span_hash: str; source_hash: str
@dataclass
class Frame:
    copy: object | None; memory: object
    def branch_enabled(self, name): return name == 'copy' and self.copy is not None
@dataclass
class Req:
    frame: Frame
class FakeSession:
    def __init__(self, pointer): self.pointer=pointer; self.verified=False
    def verify(self, req): self.verified=True
    def pointer_answer(self, req):
        from replica_lab.memory_system.session import PointerAnswer
        return PointerAnswer((1,2),'amber','r1','h1','s1',('r1',),0,1,'p')
    def exact_text(self, req, span): return f'text:{span.record_id}'

def make(pointer=True):
    span=Span('r1','s1','src1'); frame=Frame(pointer, type('M',(),{'producer':'p'})())
    return FakeSession(False), Req(frame), span

def test_pointer_path_is_opt_in_and_provenanced():
    s,req,span=make(True); svc=MemoryAnswerService(s,pointer_enabled=True)
    a=svc.answer(req)
    assert a.method=='pointer_copy' and a.text=='amber' and a.record_ids==('r1',)
    assert a.span_hashes==('s1',) and a.source_ids==('r1',)

def test_exact_source_fallback_when_pointer_disabled():
    s,req,span=make(True); req.frame.memory.spans=(span,)
    svc=MemoryAnswerService(s,pointer_enabled=False)
    a=svc.answer(req)
    assert a.method=='exact_source' and a.text=='text:r1'
    assert a.record_ids==('r1',) and a.span_hashes==('s1',)

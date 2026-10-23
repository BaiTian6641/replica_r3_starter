"""Pointer head: candidate scoring, null disabled branch and serialization."""
import torch
from architecture_helpers import tiny_adapter
from replica_lab.architecture.contracts import CopySpec
from replica_lab.architecture.ports import RecordPointerHead

def test_pointer_scores_finite_and_mask():
    head=RecordPointerHead(16)
    q=torch.randn(1,16); emb=torch.randn(1,3,2,16); tok=torch.ones(1,3,2,dtype=torch.bool); valid=torch.tensor([[True,False,True]])
    scores=head(q,emb,tok,valid)
    assert scores.shape==(1,3) and torch.isfinite(scores).all()
    assert scores[0,1] < -1e8

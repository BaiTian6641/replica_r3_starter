"""Candidate-context bias for pointer answers.

The features are derived only from source/query positions and tokenizer tokens;
no expected answer or generated text enters them. The head is a separate small
neural candidate and never overrides source eligibility or success authority.
"""
from __future__ import annotations
import torch
from torch import Tensor, nn

FEATURES = ('relative_start', 'relative_end', 'after_is', 'length_norm',
            'distance_from_is')


class CandidateContextHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(len(FEATURES), 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[-1] != len(FEATURES):
            raise ValueError(f'features must be [batch, candidates, {len(FEATURES)}]')
        if not features.is_floating_point() or not bool(torch.isfinite(features).all()):
            raise ValueError('features must be finite floating point')
        return self.linear(features).squeeze(-1)


def context_features(session, request, candidates, token_mask, provenance,
                     *, max_span: int) -> Tensor:
    """Build bounded candidate-position features for the top selected record."""
    frame = request.frame
    if not frame.memory.spans:
        raise ValueError('context features require selected memory spans')
    top = frame.memory.spans[0].record_id
    ids = [token for span in frame.memory.spans if span.record_id == top
           for token in span.token_ids]
    tokens = [t.replace('Ġ', '').replace('▁', '').strip().lower()
              for t in session.tokenizer.convert_ids_to_tokens(ids)]
    is_positions = [i for i, t in enumerate(tokens) if t in ('is', 'are', 'was', 'were')]
    rows = []
    for start, length, _ in provenance:
        last_is = max([i for i in is_positions if i < start], default=-1)
        rows.append([
            start / max(1, len(ids)),
            (start + length) / max(1, len(ids)),
            1.0 if last_is == start - 1 else 0.0,
            length / max(1, max_span),
            (start - last_is) / max(1, len(ids)) if last_is >= 0 else 1.0,
        ])
    return torch.tensor([rows], dtype=torch.float32,
                        device=frame.sense.prompt_ids.device)

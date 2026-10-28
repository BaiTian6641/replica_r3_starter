"""CPU checks for the DC-P1 v2 response objective and mask audit."""
from __future__ import annotations

import pytest
import torch
from replica_lab.dc_response_objective import (
    CLAIM_GLOBAL_RESCALE, CLAIM_SPAN_SPECIFIC, answer_span_objective,
    audit_objective_signal, reference_objective, shift_labels)


def _logits(target: int, vocab: int = 32) -> torch.Tensor:
    rows = torch.full((2, vocab), -8.0)
    rows[0, target] = 8.0
    rows[1, 1] = 8.0
    return rows


def test_multi_token_prompt_response_alignment_and_eos_supervision():
    prompt = [10, 11, 12]
    response = [20, 1]
    shifted, masks = shift_labels(
        prompt_ids=prompt, response_ids=response, answer_span=(0, 0))
    assert shifted == response
    response_start = len(prompt) - 1
    assert response_start == 2
    assert response_start + len(masks.response_mask) == (
        len(prompt) + len(response) - 1)
    assert masks.response_mask == (True, True)
    assert masks.answer_span_mask == (True, True)
    assert masks.masks_coincide is True

    logits = _logits(20)
    control = answer_span_objective(
        logits, torch.tensor(shifted), masks, answer_span_ce_weight=0.0)
    treatment = answer_span_objective(
        logits, torch.tensor(shifted), masks, answer_span_ce_weight=0.5)
    assert control["loss"].item() == treatment["full_response_ce"]
    assert treatment["loss"].item() == control["loss"].item() * 1.5
    assert treatment["applied_term"] == CLAIM_GLOBAL_RESCALE

    reference = reference_objective(
        logits.tolist(), shifted, masks, answer_span_ce_weight=0.5)
    assert reference["loss"] == pytest.approx(
        treatment["loss"].item(), rel=0.03)
    audit = audit_objective_signal(
        masks=masks, answer_span_ce_weight=0.5, logits=logits,
        label_ids=torch.tensor(shifted))
    assert audit["claim"] == CLAIM_GLOBAL_RESCALE
    assert audit["added_term_changes_learning_signal"] is False


def test_answer_span_subset_is_not_mislabeled_as_global_rescale():
    prompt = [10, 11, 12]
    response = [20, 21, 1]
    _, masks = shift_labels(
        prompt_ids=prompt, response_ids=response, answer_span=(1, 1))
    assert masks.response_mask == (True, True, True)
    assert masks.answer_span_mask == (False, True, True)
    assert masks.masks_coincide is False
    audit = audit_objective_signal(
        masks=masks, answer_span_ce_weight=0.5, logits=None, label_ids=None)
    assert audit["claim"] == CLAIM_SPAN_SPECIFIC
    assert audit["added_term_changes_learning_signal"] is True

"""Teacher-forced answer/EOS likelihood and candidate-margin regressions."""
import torch

from replica_lab.contextual_completion_objective_v1 import (
    CompletionObjectiveSpec, encode_complete_response, stop_policy_identity)
from replica_lab.intrinsic_memory_acceptance_v15 import (
    sequence_likelihood_parts, sequence_margin)


def spec():
    return CompletionObjectiveSpec(12, 16, 1, 2, (1,), stop_policy_identity((1,)),
                                   "a" * 64, "b" * 64, "none")


def test_each_candidate_uses_own_tokens_with_separate_eos_and_differentiable_margin():
    objective = spec()
    prompt = [3, 5]
    expected_encoding = encode_complete_response(prompt, [7, 9], spec=objective)
    alternative_encoding = encode_complete_response(prompt, [8, 10], spec=objective)
    logits = torch.full((1, 16, 12), -10.0, requires_grad=True)
    with torch.no_grad():
        logits[0, :, 7] = 0.0
        logits[0, :, 9] = -0.5
        logits[0, :, 8] = -2.0
        logits[0, :, 10] = -3.0
        logits[0, :, 1] = -1.0
    expected = sequence_likelihood_parts(logits, expected_encoding, spec=objective)
    alternative_logits = logits.detach().clone().requires_grad_(True)
    with torch.no_grad():
        alternative_logits[0, :, 1] = -2.0
    alternative = sequence_likelihood_parts(alternative_logits, alternative_encoding, spec=objective)
    probabilities = torch.log_softmax(logits, dim=-1)
    labels = expected_encoding.labels[0]
    terminal = expected_encoding.metadata["terminal_prediction_position"]
    assert torch.allclose(expected["answer_log_likelihood"],
                          probabilities[0, 1:terminal].gather(
                              1, labels[1:terminal, None]).sum())
    assert torch.allclose(expected["eos_log_likelihood"],
                          probabilities[0, terminal, labels[terminal]])
    assert expected["answer_tokens"] == 2 and expected["supervised_tokens"] == 3
    margin = sequence_margin(expected, alternative)
    assert margin["answer_log_likelihood"] > 0
    assert margin["eos_log_likelihood"] > 0
    assert margin["total_log_likelihood"] > 0
    margin["total_log_likelihood"].backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0

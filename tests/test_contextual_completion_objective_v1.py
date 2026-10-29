"""Synthetic CPU objective coverage; no native/development answer data is read."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
import torch

from replica_lab.contextual_completion_objective_v1 import (
    CompletionObjectiveError, CompletionObjectiveSpec, complete_response_loss,
    encode_complete_response, stop_policy_identity, validate_complete_response,
)


def spec(**changes):
    values = dict(vocab_size=32, context_tokens=8, eos_token_id=1, pad_token_id=2,
        allowed_stop_token_ids=(1, 7), allowed_stop_identity=stop_policy_identity((1, 7)),
        tokenizer_identity=hashlib.sha256(b"synthetic-tokenizer-v1").hexdigest(),
        template_identity=hashlib.sha256(b"synthetic-template-v1").hexdigest(), thinking_mode="none")
    values.update(changes)
    return CompletionObjectiveSpec(**values)


@pytest.mark.parametrize("answer", [[5], [5, 6, 8]])
def test_one_and_multiple_word_tokens_include_one_eos_with_correct_next_token_positions(answer):
    objective = spec()
    encoded = encode_complete_response([3, 4], answer, spec=objective)
    sequence = [3, 4] + answer + [1]
    assert encoded.input_ids.tolist() == [sequence + [2]*(8-len(sequence))]
    assert encoded.attention_mask.tolist() == [[1]*len(sequence)+[0]*(8-len(sequence))]
    expected = [-100] + answer + [1] + [-100]*(8-len(answer)-2)
    assert encoded.labels.tolist() == [expected]
    assert int((encoded.labels == 1).sum()) == 1
    assert int((encoded.labels == 2).sum()) == 0
    assert encoded.labels[0, len(sequence)-1:].eq(-100).all()
    counts = validate_complete_response(encoded, spec=objective)
    assert counts == {"supervised_tokens": len(answer)+1, "word_tokens": len(answer), "terminal_tokens": 1}


def test_prompt_special_tokens_are_context_not_supervised_targets():
    encoded = encode_complete_response([1, 3, 7, 4], [5], spec=spec())
    assert encoded.labels[0, :3].eq(-100).all()
    assert encoded.labels[0, 3:5].tolist() == [5, 1]


def test_perfect_word_wrong_eos_has_terminal_loss_and_live_terminal_gradient():
    objective = spec()
    encoded = encode_complete_response([3, 4], [5], spec=objective)
    raw = torch.full((1, 8, 32), -20.0)
    raw[0, 1, 5] = 20.0  # correct first answer word
    raw[0, 2, 6] = 20.0  # wrong continuation instead of EOS
    logits = raw.requires_grad_()
    result = complete_response_loss(logits, encoded, spec=objective)
    assert result.word_loss.item() < 1e-6
    assert result.terminal_loss.item() > 30
    assert result.loss.item() > 15
    result.loss.backward()
    assert logits.grad[0, 2, 1].item() < -.49
    assert logits.grad[0, 2, 6].item() > .49
    assert logits.grad[0, [0, 3, 4, 5, 6, 7]].eq(0).all()
    correct = raw.clone()
    correct[0, 2, 6] = -20
    correct[0, 2, 1] = 20
    assert complete_response_loss(correct, encoded, spec=objective).loss.item() < 1e-6


def test_loss_matches_mean_ce_over_answer_plus_eos_without_second_shift():
    objective = spec()
    encoded = encode_complete_response([3], [5, 6, 8], spec=objective)
    logits = torch.randn(1, 8, 32, generator=torch.Generator().manual_seed(9), requires_grad=True)
    result = complete_response_loss(logits, encoded, spec=objective)
    expected = torch.nn.functional.cross_entropy(logits[0, :4], torch.tensor([5, 6, 8, 1]))
    assert torch.allclose(result.loss, expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(result.loss, (3*result.word_loss+result.terminal_loss)/4)
    result.loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad[0, :4].abs().sum() > 0


@pytest.mark.parametrize("damage", ["missing_eos", "wrong_terminal", "duplicate_eos", "missing_eos_label",
                                   "pad_target", "shifted_labels", "left_padding", "identity"])
def test_incomplete_or_substituted_encoding_refused_before_loss(damage):
    objective = spec()
    encoded = encode_complete_response([3, 4], [5, 6], spec=objective)
    if damage == "missing_eos": encoded.input_ids[0, 4] = 6
    if damage == "wrong_terminal": encoded.input_ids[0, 4] = 7
    if damage == "duplicate_eos": encoded.input_ids[0, 3] = 1
    if damage == "missing_eos_label": encoded.labels[0, 3] = -100
    if damage == "pad_target": encoded.labels[0, 5] = 2
    if damage == "shifted_labels": encoded = replace(encoded, labels=encoded.labels.roll(1, 1))
    if damage == "left_padding": encoded = replace(encoded, input_ids=encoded.input_ids.roll(3, 1), attention_mask=encoded.attention_mask.roll(3, 1))
    if damage == "identity": encoded.metadata["objective_sha256"] = "0"*64
    with pytest.raises(CompletionObjectiveError):
        complete_response_loss(torch.zeros(1, 8, 32, requires_grad=True), encoded, spec=objective)


@pytest.mark.parametrize("changes", [
    {"eos_token_id": None}, {"eos_token_id": -1}, {"eos_token_id": 2}, {"eos_token_id": True},
    {"allowed_stop_token_ids": ()}, {"allowed_stop_token_ids": (1, 1)},
    {"allowed_stop_token_ids": (7,)}, {"allowed_stop_identity": "0"*64},
    {"tokenizer_identity": "unbound"}, {"template_identity": ""},
    {"padding_side": "left"}, {"label_alignment": "shift_again"},
])
def test_invalid_or_missing_terminal_and_identity_bindings_refused(changes):
    with pytest.raises(CompletionObjectiveError):
        spec(**changes)


@pytest.mark.parametrize("answer", [[], [1], [5, 1], [5, 1, 1], [5, 7], [2], [-1], [32], [True]])
def test_raw_answer_must_be_nonempty_unpadded_and_unterminated(answer):
    with pytest.raises(CompletionObjectiveError):
        encode_complete_response([3, 4], answer, spec=spec())


def test_eos_capacity_is_reserved_and_never_truncated():
    objective = spec(context_tokens=4)
    encoded = encode_complete_response([3, 4], [5], spec=objective)
    assert encoded.input_ids.tolist() == [[3, 4, 5, 1]]
    with pytest.raises(CompletionObjectiveError, match="EOS is never truncated"):
        encode_complete_response([3, 4], [5, 6], spec=objective)


def test_stable_checkpoint_metadata_distinguishes_tokenizer_template_and_thinking_mode():
    objective = spec()
    encoded = encode_complete_response([3, 4], [5], spec=objective)
    metadata = json.loads(json.dumps(encoded.metadata, sort_keys=True))
    assert metadata["objective_sha256"] == spec().sha256
    assert metadata["objective_identity"] == objective.identity()
    for changed in (spec(template_identity="a"*64), spec(tokenizer_identity="b"*64), spec(thinking_mode="declared-other-mode")):
        assert changed.sha256 != objective.sha256
        with pytest.raises(CompletionObjectiveError, match="identity"):
            validate_complete_response(encoded, spec=changed)


def test_loss_rejects_nonfinite_logits_and_wrong_vocabulary():
    encoded = encode_complete_response([3, 4], [5], spec=spec())
    for logits in (torch.zeros(1, 8, 31), torch.full((1, 8, 32), float("nan"))):
        with pytest.raises(CompletionObjectiveError, match="finite logits"):
            complete_response_loss(logits, encoded, spec=spec())


def test_live_contextual_producer_reader_eos_objective_one_optimizer_integration():
    from replica_lab.architecture.intrinsic_memory_layout_v4 import IntrinsicMemoryLayout
    from replica_lab.contextual_memory import canonical_spans
    from replica_lab.contextual_memory_layout_v4 import ContextualMemoryLayoutV4
    from replica_lab.contextual_spark_fixture import build_contextual_adapter
    from replica_lab.optimizers import NativeFusedAdamW
    from replica_lab.tiered_memory import MemoryScope
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        adapter, *_ = build_contextual_adapter(seed=17, layers=8, spine_sites=(3, 7), spine_slots=8, memory_size=256)
        path = ContextualMemoryLayoutV4(adapter, IntrinsicMemoryLayout(8, decoder_layers=8),
                                       source_identity="synthetic_completion_CPU_fixture")
        path.adapter.extension.spine = None
        objective = spec()
        encoded = encode_complete_response([3, 4], [5], spec=objective)
        scope = MemoryScope("fixture", "main", "assistant", as_of=10)
        spans = canonical_spans("Synthetic badge rule.", record_id="rule", source_id="fixture",
                                source_epoch=1, available_at=1)
        producer_gradients = []
        for _ in range(2):
            path.optimizer.zero_grad(set_to_none=True)
            live = path.training_values(spans, scope=scope)
            logits = path.forward(encoded.input_ids, encoded.attention_mask, live, scope=scope)
            logits.retain_grad()
            losses = complete_response_loss(logits, encoded, spec=objective)
            losses.loss.backward()
            terminal = encoded.metadata["terminal_prediction_position"]
            assert logits.grad[0, terminal, 1] < 0
            assert all(torch.isfinite(p.grad).all() for group in path.optimizer.param_groups
                       for p in group["params"] if p.grad is not None)
            producer_gradients.append(sum(float(p.grad.abs().sum()) for p in path.encoder.parameters() if p.grad is not None))
            assert all(port.output_projection.weight.grad.abs().sum() > 0 for port in path.reader.ports.values())
            path.optimizer.step()
        assert producer_gradients[0] == 0 and producer_gradients[1] > 0
        assert isinstance(path.optimizer, NativeFusedAdamW)
        owned = [id(p) for group in path.optimizer.param_groups for p in group["params"]]
        assert len(owned) == len(set(owned))
        assert path.adapter.base_unchanged()
        assert all(p.grad is None for p in path.adapter.base.parameters())
    finally:
        torch.set_num_threads(old_threads)


def test_cpu_protocol_binds_current_helper_and_preserves_native_claim_boundary():
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "configs/contextual_completion_objective_protocol_v1.json").read_text())
    assert protocol["status"] == "CPU_candidate_only_not_native_authorized"
    assert protocol["native_semantic_utility"] == "unknown"
    assert protocol["padding_side"] == "right"
    for filename, digest in protocol["code_bindings"].items():
        assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == digest, filename


def test_decoder_pad_stop_is_preserved_without_becoming_a_training_target():
    objective = spec(allowed_stop_token_ids=(1, 2), allowed_stop_identity=stop_policy_identity((1, 2)))
    encoded = encode_complete_response([3, 4], [5], spec=objective)
    assert encoded.metadata["objective_identity"]["allowed_stop_token_ids"] == [1, 2]
    assert encoded.metadata["objective_identity"]["allowed_stop_identity"] == stop_policy_identity((1, 2))
    assert objective.identity()["training_terminal_identity"] == spec().identity()["training_terminal_identity"]
    assert encoded.labels[0, 1:3].tolist() == [5, 1]
    assert not (encoded.labels == 2).any()
    with pytest.raises(CompletionObjectiveError):
        encode_complete_response([3, 4], [5, 2], spec=objective)
    wrong_terminal = replace(encoded, input_ids=encoded.input_ids.clone())
    wrong_terminal.input_ids[0, 3] = 2
    with pytest.raises(CompletionObjectiveError, match="terminal/PAD"):
        validate_complete_response(wrong_terminal, spec=objective)
    raw = torch.full((1, 8, 32), -20.0)
    raw[0, 1, 5] = 20.0
    raw[0, 2, 2] = 20.0  # permitted decoder stop, incorrect EOS training target
    logits = raw.requires_grad_()
    result = complete_response_loss(logits, encoded, spec=objective)
    assert result.word_loss.item() < 1e-6 and result.terminal_loss.item() > 30
    result.loss.backward()
    assert logits.grad[0, 2, 1] < -.49 and logits.grad[0, 2, 2] > .49

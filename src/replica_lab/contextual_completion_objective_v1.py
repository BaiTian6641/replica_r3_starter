"""Complete-response objective candidate, independent from sealed training runs.

Inputs remain right padded and use the existing adapter ABI: logit position i
predicts input token i+1. Raw answer token IDs must exclude terminal tokens; this
encoder appends the explicitly bound EOS once. Prompt and PAD are never targets.
PAD may remain in the separately bound decoder stop inventory without changing
this EOS-only training terminal policy.
This fixes an objective-coverage gap, not demonstrated native behavior or the
cause of a wrong first answer word/repetition. Future native use needs its own
protocol, budget, authorization and fresh evaluation worlds.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import hashlib
import json

import torch
import torch.nn.functional as F

OBJECTIVE = "replica.contextual_complete_response.objective.v1"
ENCODING = "replica.contextual_complete_response.encoding.v1"
IGNORE_INDEX = -100


class CompletionObjectiveError(ValueError):
    pass


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def stop_policy_identity(allowed_stop_token_ids):
    """Decoder stop inventory identity, independent of the training terminal.

    PAD may remain a decoder stop. That does not make it a training target.
    Callers must bind this inventory to their existing generation protocol.
    """
    return _digest({"generation_stop_token_ids": list(allowed_stop_token_ids)})


@dataclass(frozen=True)
class CompletionObjectiveSpec:
    vocab_size: int
    context_tokens: int
    eos_token_id: int
    pad_token_id: int
    allowed_stop_token_ids: tuple[int, ...]
    allowed_stop_identity: str
    tokenizer_identity: str
    template_identity: str
    thinking_mode: str
    padding_side: str = "right"
    label_alignment: str = "logit_i_predicts_token_i_plus_1_no_additional_shift"

    def __post_init__(self):
        if (type(self.vocab_size) is not int or self.vocab_size < 3
                or type(self.context_tokens) is not int or self.context_tokens < 3
                or any(type(token) is not int or not 0 <= token < self.vocab_size
                       for token in (self.eos_token_id, self.pad_token_id))
                or self.eos_token_id == self.pad_token_id
                or not isinstance(self.allowed_stop_token_ids, tuple)
                or not self.allowed_stop_token_ids
                or any(type(token) is not int or not 0 <= token < self.vocab_size
                       for token in self.allowed_stop_token_ids)
                or tuple(sorted(set(self.allowed_stop_token_ids))) != self.allowed_stop_token_ids
                or self.eos_token_id not in self.allowed_stop_token_ids):
            raise CompletionObjectiveError("invalid vocabulary, context, bound EOS/PAD or allowed stop inventory")
        if (not all(_sha256(value) for value in (self.tokenizer_identity, self.template_identity,
                                                self.allowed_stop_identity))
                or self.allowed_stop_identity != stop_policy_identity(self.allowed_stop_token_ids)
                or not isinstance(self.thinking_mode, str) or not self.thinking_mode
                or self.padding_side != "right"
                or self.label_alignment != "logit_i_predicts_token_i_plus_1_no_additional_shift"):
            raise CompletionObjectiveError("objective/tokenizer/template/stop identity or alignment differs")

    def identity(self):
        payload = {"format": OBJECTIVE, **asdict(self), "ignore_index": IGNORE_INDEX,
                   "loss": "mean_cross_entropy_over_answer_tokens_plus_one_EOS",
                   "training_terminal_identity": _digest({"rule": "append_bound_EOS_exactly_once",
                                                         "eos_token_id": self.eos_token_id})}
        payload["allowed_stop_token_ids"] = list(self.allowed_stop_token_ids)
        return payload

    @property
    def sha256(self):
        return _digest(self.identity())


def _token_ids(values, *, spec, name):
    if (not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values
            or any(type(token) is not int or not 0 <= token < spec.vocab_size for token in values)
            or spec.pad_token_id in values):
        raise CompletionObjectiveError(f"{name} must contain nonempty valid unpadded token IDs")
    return list(values)


@dataclass(frozen=True)
class CompleteResponseEncoding:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    metadata: dict


def encode_complete_response(prompt_ids, answer_ids, *, spec: CompletionObjectiveSpec, device="cpu"):
    """Append bound EOS, right pad and place already shifted response-only labels.

    Tokenization/rendering occur upstream under the identities in spec. Neither
    tokenizer defaults nor add_eos_token behavior can add an implicit terminal.
    Empty responses and raw answers already containing any allowed stop refuse.
    """
    if not isinstance(spec, CompletionObjectiveSpec):
        raise CompletionObjectiveError("an explicit completion objective specification is required")
    prompt = _token_ids(prompt_ids, spec=spec, name="prompt")
    answer = _token_ids(answer_ids, spec=spec, name="answer")
    if any(token in spec.allowed_stop_token_ids for token in answer):
        raise CompletionObjectiveError("raw answer already contains a terminal; duplicate or substituted EOS refused")
    response = answer + [spec.eos_token_id]
    sequence = prompt + response
    if len(sequence) > spec.context_tokens:
        raise CompletionObjectiveError("complete response exceeds context budget; EOS is never truncated")
    padding = spec.context_tokens-len(sequence)
    values = sequence + [spec.pad_token_id]*padding
    mask = [1]*len(sequence) + [0]*padding
    labels = [IGNORE_INDEX]*spec.context_tokens
    begin = len(prompt)-1
    labels[begin:begin+len(response)] = response
    metadata = {"format": ENCODING, "objective_identity": spec.identity(), "objective_sha256": spec.sha256,
                "prompt_tokens": len(prompt), "answer_tokens": len(answer),
                "sequence_tokens": len(sequence), "supervised_tokens": len(response),
                "terminal_prediction_position": len(sequence)-2,
                "terminal_input_position": len(sequence)-1,
                "input_tokens_sha256": _digest(values)}
    encoding = CompleteResponseEncoding(torch.tensor([values], dtype=torch.long, device=device),
        torch.tensor([mask], dtype=torch.long, device=device),
        torch.tensor([labels], dtype=torch.long, device=device), metadata)
    validate_complete_response(encoding, spec=spec)
    return encoding


def validate_complete_response(encoding, *, spec: CompletionObjectiveSpec):
    """Refuse incomplete/tampered terminal coverage before loss evaluation."""
    if not isinstance(encoding, CompleteResponseEncoding) or not isinstance(spec, CompletionObjectiveSpec):
        raise CompletionObjectiveError("versioned complete-response encoding and objective required")
    tensors = (encoding.input_ids, encoding.attention_mask, encoding.labels)
    if any(not torch.is_tensor(value) or value.dtype != torch.long or value.shape != (1, spec.context_tokens)
           or value.device != tensors[0].device for value in tensors):
        raise CompletionObjectiveError("complete-response tensor layout differs")
    metadata = encoding.metadata
    if (not isinstance(metadata, dict) or metadata.get("format") != ENCODING
            or metadata.get("objective_identity") != spec.identity()
            or metadata.get("objective_sha256") != spec.sha256):
        raise CompletionObjectiveError("objective/tokenizer/template identity differs")
    prompt, answer = metadata.get("prompt_tokens"), metadata.get("answer_tokens")
    if type(prompt) is not int or type(answer) is not int or prompt < 1 or answer < 1:
        raise CompletionObjectiveError("complete prompt/answer token accounting required")
    sequence = prompt+answer+1
    if (sequence > spec.context_tokens or metadata.get("sequence_tokens") != sequence
            or metadata.get("supervised_tokens") != answer+1
            or metadata.get("terminal_prediction_position") != sequence-2
            or metadata.get("terminal_input_position") != sequence-1):
        raise CompletionObjectiveError("terminal coverage/token accounting differs")
    # Inputs/labels are discrete, nontrainable data. Identity checks on these do
    # not detach logits, contextual values, producer tensors or reader weights.
    ids = encoding.input_ids[0].tolist()
    response = ids[prompt:sequence]
    if (any(not 0 <= token < spec.vocab_size for token in ids)
            or spec.pad_token_id in ids[:sequence]
            or response[-1] != spec.eos_token_id
            or any(token in spec.allowed_stop_token_ids for token in response[:-1])
            or ids[sequence:] != [spec.pad_token_id]*(spec.context_tokens-sequence)):
        raise CompletionObjectiveError("missing, duplicate or invalid terminal/PAD in complete response")
    expected_labels = [IGNORE_INDEX]*spec.context_tokens
    expected_labels[prompt-1:sequence-1] = response
    if (encoding.labels[0].tolist() != expected_labels
            or encoding.attention_mask[0].tolist() != [1]*sequence+[0]*(spec.context_tokens-sequence)
            or metadata.get("input_tokens_sha256") != _digest(ids)):
        raise CompletionObjectiveError("response labels, right padding or input identity differ")
    return {"supervised_tokens": answer+1, "word_tokens": answer, "terminal_tokens": 1}


@dataclass(frozen=True)
class CompletionLoss:
    loss: torch.Tensor
    word_loss: torch.Tensor
    terminal_loss: torch.Tensor
    supervised_tokens: int


def complete_response_loss(logits, encoding, *, spec: CompletionObjectiveSpec):
    """Differentiable CE; labels are already next-token aligned, never shift again."""
    counts = validate_complete_response(encoding, spec=spec)
    if (not torch.is_tensor(logits) or not logits.is_floating_point()
            or logits.shape != (1, spec.context_tokens, spec.vocab_size)
            or logits.device != encoding.labels.device or not torch.isfinite(logits).all()):
        raise CompletionObjectiveError("finite logits matching the bound vocabulary/context are required")
    losses = F.cross_entropy(logits.float().flatten(0, 1), encoding.labels.flatten(),
                             ignore_index=IGNORE_INDEX, reduction="none")
    begin = encoding.metadata["prompt_tokens"]-1
    end = encoding.metadata["terminal_prediction_position"]
    word = losses[begin:end]
    terminal = losses[end]
    return CompletionLoss(loss=(word.sum()+terminal)/counts["supervised_tokens"],
                          word_loss=word.mean(), terminal_loss=terminal,
                          supervised_tokens=counts["supervised_tokens"])

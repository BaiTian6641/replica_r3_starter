"""Self-contained deterministic tokenizer fixtures for CPU unit contracts.

This is not the sealed MiniCPM tokenizer and is never used for native evidence.
It provides reversible text/ID mappings for orchestration and tiny-model tests
without ignored model metadata or downloads. Genuine tokenizer parity tests
must use ``require_local_artifacts`` instead.
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence


class DeterministicTinyTokenizer:
    """A small reversible tokenizer with the methods used by test backends."""

    def __init__(self, *, phrases: Iterable[str] = (), texts: Iterable[str] = ()) -> None:
        self.is_fast = True
        self._token_to_id = {"<pad>": 0, "<eos>": 1, "<unk>": 2, "<s>": 3}
        self._id_to_token = {value: key for key, value in self._token_to_id.items()}
        self._special_ids = frozenset(self._id_to_token)
        self._phrase_ids: dict[str, int] = {}
        for phrase in sorted(set(phrases)):
            self._phrase_ids[phrase] = self._add(phrase)
        for text in sorted(set(texts)):
            for piece in self._pieces(text):
                self._add(piece)

    @staticmethod
    def _pieces(text: str) -> list[str]:
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

    def _add(self, token: str) -> int:
        if token not in self._token_to_id:
            token_id = len(self._token_to_id) + 12
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token
        return self._token_to_id[token]

    def __call__(self, text: str, *, add_special_tokens: bool = False,
                 **_: object) -> dict[str, list[int]]:
        if text in self._phrase_ids:
            return {"input_ids": [self._phrase_ids[text]]}
        return {"input_ids": [self._add(piece) for piece in self._pieces(text)]}

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True,
               clean_up_tokenization_spaces: bool = False) -> str:
        del clean_up_tokenization_spaces
        pieces = []
        for value in ids:
            token_id = int(value)
            if skip_special_tokens and token_id in self._special_ids:
                continue
            if token_id in self._id_to_token:
                pieces.append(self._id_to_token[token_id])
            elif 0 <= token_id <= 0x10FFFF and not 0xD800 <= token_id <= 0xDFFF:
                pieces.append(chr(token_id))
            else:
                pieces.append("�")
        return " ".join(pieces)

    def apply_chat_template(self, messages, *, tokenize: bool = False,
                            add_generation_prompt: bool = True,
                            enable_thinking: bool = False, **_: object):
        del add_generation_prompt, enable_thinking
        rendered = str(messages[-1]["content"])
        if tokenize:
            return self(rendered)["input_ids"]
        return rendered


def tiny_tokenizer(*, phrases: Iterable[str] = (),
                   texts: Iterable[str] = ()) -> DeterministicTinyTokenizer:
    return DeterministicTinyTokenizer(phrases=phrases, texts=texts)

"""Byte-level BPE tokenizer for RoyLM-1.

RoyLM-0 used a character vocabulary: every symbol was one token, so
"hello world" became 11 tokens (h e l l o _ w o r l d).

RoyLM-1 trains a byte-level Byte-Pair-Encoding tokenizer -- the same family
GPT-2 uses. It learns common chunks from the data, so "hello world" becomes
a couple of subword tokens instead. The lesson: the tokenizer is part of the
model pipeline. It is trained, saved, and shipped alongside the weights.

"Byte-level" means the base alphabet is the 256 possible bytes, so any text
(including emoji or unseen characters) can be encoded with no <unk> token.
"""
from __future__ import annotations

import os

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .config import EOT_TOKEN


class BPETokenizer:
    """Thin wrapper around a Hugging Face `tokenizers` byte-level BPE model.

    One object owns the entire text<->token-id mapping, so `prepare.py`,
    `train.py`, and `sample.py` all encode/decode identically.
    """

    def __init__(self, tokenizer: Tokenizer):
        self._tok = tokenizer
        # id of the story-separator token; falls back to 0 if it's absent
        eot = tokenizer.token_to_id(EOT_TOKEN)
        self.eot_id = eot if eot is not None else 0

    @classmethod
    def train(cls, files, vocab_size, save_path=None, eot_token=EOT_TOKEN):
        """Train a byte-level BPE tokenizer on one or more text files."""
        if isinstance(files, str):
            files = [files]

        tok = Tokenizer(models.BPE(unk_token=None))
        # ByteLevel pre-tokenizer + matching decoder make the round trip lossless.
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=[eot_token],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )
        tok.train(files, trainer)

        wrapper = cls(tok)
        if save_path is not None:
            wrapper.save(save_path)
        return wrapper

    @classmethod
    def load(cls, path):
        return cls(Tokenizer.from_file(path))

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._tok.save(path)

    def encode(self, text: str):
        """text -> list[int] of token ids."""
        return self._tok.encode(text).ids

    def decode(self, ids):
        """list[int] of token ids -> text (special tokens dropped)."""
        return self._tok.decode(list(ids))

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

"""Encode TinyStories text into token-id arrays: train.bin, val.bin, meta.json.

    python data/tinystories/prepare.py

Each story is encoded with the trained BPE tokenizer and terminated with the
<|endoftext|> token so the model learns where stories begin and end. Token ids
are written as uint16 (the 4,096-token vocab fits comfortably under 65,536),
which is what train.py memory-maps.
"""
import json
import os
import pathlib
import sys

import numpy as np

# Make the `roylm` package importable when this script is run directly.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from roylm.config import (EOT_TOKEN, META_PATH, TOKENIZER_PATH, TRAIN_BIN,
                          TRAIN_TXT, VAL_BIN, VAL_TXT)
from roylm.tokenizer_bpe import BPETokenizer


def read_stories(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # TinyStories separates stories with <|endoftext|>. If that marker is
    # absent (e.g. a custom corpus), treat the whole file as one stream.
    if EOT_TOKEN in text:
        return [s for s in text.split(EOT_TOKEN) if s.strip()]
    return [text]


def write_bin(tok, stories, bin_path):
    """Encode stories and stream them to a .bin file as uint16 token ids."""
    n_tokens = 0
    with open(bin_path, "wb") as out:
        for s in stories:
            ids = tok.encode(s)
            ids.append(tok.eot_id)
            np.asarray(ids, dtype=np.uint16).tofile(out)
            n_tokens += len(ids)
    return n_tokens


def main():
    if not os.path.exists(TOKENIZER_PATH):
        raise SystemExit("No tokenizer.json. Run train_tokenizer.py first.")
    if not os.path.exists(TRAIN_TXT):
        raise SystemExit("No TinyStories text found. Run download.py first.")

    tok = BPETokenizer.load(TOKENIZER_PATH)
    assert tok.vocab_size <= 65536, "vocab too large for uint16 storage"

    # Sanity check: byte-level BPE round-trips text exactly.
    sample = "Once upon a time, there was a cat who liked to nap."
    assert tok.decode(tok.encode(sample)) == sample, "encode/decode round trip failed"
    print("encode/decode round trip OK")

    train_stories = read_stories(TRAIN_TXT)
    if os.path.exists(VAL_TXT):
        val_stories = read_stories(VAL_TXT)
    else:
        # No dedicated validation file: hold out the last 10% of stories.
        cut = max(1, int(len(train_stories) * 0.1))
        val_stories = train_stories[-cut:]
        train_stories = train_stories[:-cut]

    print(f"stories: {len(train_stories):,} train / {len(val_stories):,} val")

    n_train = write_bin(tok, train_stories, TRAIN_BIN)
    print(f"train: {n_train:,} tokens -> {os.path.relpath(TRAIN_BIN)}")
    n_val = write_bin(tok, val_stories, VAL_BIN)
    print(f"val:   {n_val:,} tokens -> {os.path.relpath(VAL_BIN)}")

    with open(META_PATH, "w") as f:
        json.dump({"vocab_size": tok.vocab_size,
                   "eot_id": tok.eot_id,
                   "tokenizer_path": os.path.relpath(TOKENIZER_PATH)}, f, indent=2)
    print(f"wrote meta -> {os.path.relpath(META_PATH)}")


if __name__ == "__main__":
    main()

"""Train the byte-level BPE tokenizer on TinyStories -> tokenizer.json.

    python data/tinystories/train_tokenizer.py
    python data/tinystories/train_tokenizer.py --vocab-size 8192

This is the step that did not exist in RoyLM-0. Instead of reading off the set
of characters in the data, we *learn* a vocabulary of common subword chunks.
"""
import argparse
import os
import pathlib
import sys

# Make the `roylm` package importable when this script is run directly.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from roylm.config import TOKENIZER_PATH, TRAIN_TXT, VAL_TXT, VOCAB_SIZE
from roylm.tokenizer_bpe import BPETokenizer


def main():
    ap = argparse.ArgumentParser(description="Train a BPE tokenizer on TinyStories")
    ap.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    ap.add_argument("--out", default=TOKENIZER_PATH)
    args = ap.parse_args()

    files = [p for p in (TRAIN_TXT, VAL_TXT) if os.path.exists(p)]
    if not files:
        raise SystemExit("No TinyStories text found. Run download.py first.")

    print(f"training byte-level BPE (target vocab {args.vocab_size}) on:")
    for p in files:
        print(f"  {os.path.relpath(p)} ({os.path.getsize(p) / 1e6:.1f} MB)")

    tok = BPETokenizer.train(files, args.vocab_size, save_path=args.out)
    print(f"final vocab size: {tok.vocab_size}")
    print(f"saved tokenizer -> {os.path.relpath(args.out)}")

    # Show what subword tokenization looks like on a sample sentence.
    demo = "Once upon a time, there was a little robot who loved to paint."
    ids = tok.encode(demo)
    print(f"\ndemo: {demo!r}")
    print(f"  {len(demo)} chars -> {len(ids)} tokens")
    print(f"  ids: {ids}")
    print(f"  round trip: {tok.decode(ids)!r}")


if __name__ == "__main__":
    main()

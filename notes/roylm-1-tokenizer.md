# RoyLM-1 tokenizer notes

## From characters to subwords

RoyLM-0 used a **character** vocabulary. It built `stoi` / `itos` by taking the
sorted set of characters in the data (~65 of them), so every symbol was one
token:

```
"hello world"  ->  h e l l o _ w o r l d        (11 tokens)
```

RoyLM-1 uses a **byte-level BPE** (Byte-Pair Encoding) tokenizer, the same
family GPT-2 uses. BPE starts from individual bytes and repeatedly merges the
most frequent adjacent pair into a new token. After training on TinyStories it
learns common chunks:

```
t, h, e  ->  th  ->  the
i, n, g  ->  ing
" ", o, n, c, e  ->  " once"
```

So a word like `"hello"` may become a single token, and `"hello world"` becomes
just a couple of tokens instead of 11.

## Why this matters

1. **Shorter sequences.** Fewer tokens per sentence means each 256-token context
   window covers far more text than 256 characters did.
2. **The model predicts tokens, not letters.** This is how real LLMs work. The
   output layer now spans 4,096 possible next tokens instead of ~65 characters
   (`[B, T, 4096]` logits).
3. **The tokenizer is part of the model.** It is *trained* on data, *saved*
   (`tokenizer.json`), and must be *shipped with the weights*. Encode and decode
   must use the exact same tokenizer the model was trained on, or the ids mean
   nothing. This is the central RoyLM-1 lesson.

## Choices made here

- **Byte-level** base alphabet (256 bytes). Every possible input maps to known
  tokens, so there is **no `<unk>` token** and the encode/decode round trip is
  lossless -- even for characters never seen in training.
- **`vocab_size = 4096`.** Small enough to train and store on a laptop (ids fit
  in `uint16`), large enough to capture useful subwords. RoyLM-0 had ~65.
- **`<|endoftext|>`** is a special token appended after every story so the model
  learns where stories start and end, and so sampling can stop cleanly.

## Files

- `roylm/tokenizer_bpe.py` -- the `BPETokenizer` wrapper (train / load / save /
  encode / decode). One object owns the whole text<->id mapping.
- `data/tinystories/train_tokenizer.py` -- trains it and writes `tokenizer.json`.
- `data/tinystories/prepare.py` -- uses it to encode the corpus into `.bin`
  files.

## Retraining / experimenting

```bash
# bigger vocabulary
python data/tinystories/train_tokenizer.py --vocab-size 8192
python data/tinystories/prepare.py        # re-encode with the new tokenizer
```

If you change the vocabulary you must re-run `prepare.py` (the ids change) and
retrain the model (the embedding table and output head are sized to the vocab).

## Things to look at

- Run `train_tokenizer.py` and read its demo output: how many tokens does a
  sentence become? Which words are single tokens vs split into pieces?
- Numbers and rare words often fragment into many tokens -- a known quirk of BPE.
- Leading spaces are part of tokens (`" world"` != `"world"`); that is the
  byte-level convention and is why decoding reproduces spacing correctly.

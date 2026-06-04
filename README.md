# RoyLM-1

RoyLM-1 is a small decoder-only GPT trained from scratch using a **byte-level
BPE tokenizer**. It moves beyond the character-level modeling of
[RoyLM-0](https://github.com/ryouol/RoyLM-0) by learning subword tokens and
training on natural story data (TinyStories).

It is not instruction-tuned and not a chatbot. It is a base language model: it
continues text in the style of its training data, one **token** at a time.

```
RoyLM-0 (character-level)   "hello world" -> h e l l o _ w o r l d   (11 tokens)
RoyLM-1 (subword/token)     "hello world" -> ["hello", " world"]      (2 tokens)
```

## What changed from RoyLM-0

| | RoyLM-0 | RoyLM-1 |
| --- | --- | --- |
| Tokenizer | character set (~65) | trained byte-level BPE |
| Vocab size | ~65 | 4,096 |
| Dataset | Tiny Shakespeare | TinyStories |
| Output logits | `[B, T, 65]` | `[B, T, 4096]` |
| Context length | 256 | 256 |
| `n_embd` / `n_layer` / `n_head` | 384 / 6 / 6 | 256 / 6 / 8 |
| Checkpoint loading | `weights_only=False` | `weights_only=True` (config stored as plain dict) |

The transformer itself (`roylm/model.py`) is the same architecture as RoyLM-0.
The real changes are the tokenizer, the dataset, and treating the tokenizer as
a trained, saved, shipped part of the pipeline.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## The pipeline

Run the data pipeline once, then train and sample. All commands are run from
the repo root.

```bash
# 1. Download TinyStories text (defaults to a ~200 MB train slice; use
#    --max-mb 0 for the full ~2 GB, or --max-mb 5 for a quick experiment)
python data/tinystories/download.py

# 2. Train the byte-level BPE tokenizer  ->  data/tinystories/tokenizer.json
python data/tinystories/train_tokenizer.py

# 3. Encode text into token ids  ->  train.bin, val.bin, meta.json
python data/tinystories/prepare.py

# 4. Train the model  ->  checkpoints/roylm-1.pt
python -m roylm.train

# 5. Generate text -- one-shot, or play interactively
python -m roylm.sample --prompt "Once upon a time" --tokens 300
python -m roylm.prompt       # type prompts in a loop; tweak settings live
```

`train.py` accepts overrides, e.g. `python -m roylm.train --max-iters 2000
--batch-size 8`. `sample.py` supports `--temperature`, `--top-k`, `--top-p`,
and `--num-samples`.

## Default configuration

Set in `roylm/config.py` (the single source of truth for paths and
hyperparameters):

| Stat | Value |
| --- | --- |
| Vocab size | 4,096 (byte-level BPE) |
| Context length (`block_size`) | 256 tokens |
| Embedding width (`n_embd`) | 256 |
| Layers (`n_layer`) | 6 |
| Attention heads (`n_head`) | 8 |
| Dropout | 0.1 |
| Batch size | 16 |
| Learning rate | 3e-4 (constant; AdamW) |
| Gradient clipping | 1.0 |
| Iterations | 10,000 |

Training reports validation **loss** and **perplexity** (`e^loss`) at every
eval interval and prints a short generated sample periodically.

## Project layout

```
roylm/
  config.py          paths + model/training hyperparameters (single source of truth)
  tokenizer_bpe.py   byte-level BPE wrapper: train / load / save / encode / decode
  model.py           decoder-only GPT (same architecture as RoyLM-0)
  train.py           training loop: perplexity, periodic samples, checkpointing
  sample.py          generation with temperature / top-k / top-p
  prompt.py          interactive REPL: type prompts, tweak /temp /topk /topp live
data/tinystories/
  download.py        fetch TinyStories text (with a size cap for laptops)
  train_tokenizer.py train the BPE tokenizer
  prepare.py         encode text -> train.bin / val.bin / meta.json
checkpoints/         trained weights (roylm-1.pt)
notes/               tokenizer notes + training log
```

Downloaded text, tokenizer, encoded `.bin` files, and checkpoints are generated
artifacts and are ignored by Git (see `.gitignore`).

## Status

Code complete and smoke-tested end to end (download -> tokenizer -> encode ->
train -> sample). No trained checkpoint is committed; run the pipeline above to
produce `checkpoints/roylm-1.pt`. See `notes/roylm-1-training-log.md` to record
runs.

## RoyLM roadmap

- **RoyLM-0** - character-level GPT on Tiny Shakespeare. Understand transformer internals.
- **RoyLM-1** - BPE token-level GPT on TinyStories. Understand the real tokenizer + training pipeline. *(this repo)*
- **RoyLM-2** - larger dataset + better training loop (LR schedules, eval, checkpointing).
- **RoyLM-3** - instruction tuning: turn the base model into a basic assistant.
- **RoyLM-4** - RAG + tool use.
- **RoyLM-5** - inference optimization (KV cache, quantization, Metal/C++).

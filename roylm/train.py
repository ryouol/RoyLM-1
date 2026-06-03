"""Train RoyLM-1: a token-level (byte-level BPE) GPT on TinyStories.

Run from the repo root after the data pipeline has produced train.bin/val.bin:

    python -m roylm.train

Any hyperparameter can be overridden from the command line, e.g.

    python -m roylm.train --max-iters 2000 --batch-size 8

The training loop is intentionally the same shape as RoyLM-0 (constant LR,
AdamW, gradient clipping). What is new: a much larger vocab, perplexity
reporting, periodic text samples, and clean checkpointing.
"""
import argparse
import json
import math
import os
import time
from dataclasses import asdict

import numpy as np
import torch

from .config import (CKPT_PATH, CHECKPOINT_DIR, GPTConfig, META_PATH,
                     TOKENIZER_PATH, TRAIN_BIN, TrainConfig, VAL_BIN)
from .model import GPT
from .tokenizer_bpe import BPETokenizer


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_args(mcfg: GPTConfig, tcfg: TrainConfig):
    ap = argparse.ArgumentParser(description="Train RoyLM-1")
    # model
    ap.add_argument("--block-size", type=int, default=mcfg.block_size)
    ap.add_argument("--n-layer", type=int, default=mcfg.n_layer)
    ap.add_argument("--n-head", type=int, default=mcfg.n_head)
    ap.add_argument("--n-embd", type=int, default=mcfg.n_embd)
    ap.add_argument("--dropout", type=float, default=mcfg.dropout)
    # training
    ap.add_argument("--batch-size", type=int, default=tcfg.batch_size)
    ap.add_argument("--learning-rate", type=float, default=tcfg.learning_rate)
    ap.add_argument("--max-iters", type=int, default=tcfg.max_iters)
    ap.add_argument("--eval-interval", type=int, default=tcfg.eval_interval)
    ap.add_argument("--eval-iters", type=int, default=tcfg.eval_iters)
    ap.add_argument("--sample-interval", type=int, default=tcfg.sample_interval)
    ap.add_argument("--seed", type=int, default=tcfg.seed)
    return ap.parse_args()


def main():
    mcfg, tcfg = GPTConfig(), TrainConfig()
    args = parse_args(mcfg, tcfg)

    device = get_device()
    print(f"device: {device}")
    torch.manual_seed(args.seed)

    if not os.path.exists(META_PATH):
        raise SystemExit(
            "meta.pkl not found. Run the data pipeline first:\n"
            "  python data/tinystories/download.py\n"
            "  python data/tinystories/train_tokenizer.py\n"
            "  python data/tinystories/prepare.py"
        )
    with open(META_PATH) as f:
        meta = json.load(f)
    vocab_size = meta["vocab_size"]

    cfg = GPTConfig(
        block_size=args.block_size,
        vocab_size=vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    print(f"vocab_size={vocab_size}  block_size={cfg.block_size}  "
          f"n_layer={cfg.n_layer}  n_head={cfg.n_head}  n_embd={cfg.n_embd}")

    # memmap keeps the (potentially large) token arrays on disk.
    train_data = np.memmap(TRAIN_BIN, dtype=np.uint16, mode="r")
    val_data = np.memmap(VAL_BIN, dtype=np.uint16, mode="r")
    print(f"train tokens: {len(train_data):,}   val tokens: {len(val_data):,}")

    def get_batch(split):
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - cfg.block_size, (args.batch_size,))
        # Targets are the inputs shifted one *token* to the right.
        x = torch.stack([torch.from_numpy(data[i:i + cfg.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + cfg.block_size].astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)

    model = GPT(cfg).to(device)
    print(f"parameters: {model.num_params() / 1e6:.2f}M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    tok = BPETokenizer.load(TOKENIZER_PATH) if os.path.exists(TOKENIZER_PATH) else None

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ("train", "val"):
            losses = torch.zeros(args.eval_iters)
            for k in range(args.eval_iters):
                x, y = get_batch(split)
                _, loss = model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    @torch.no_grad()
    def sample(prompt="Once upon a time", n=None):
        if tok is None:
            return "(no tokenizer available for sampling)"
        n = n or tcfg.sample_tokens
        model.eval()
        ids = tok.encode(prompt) or [tok.eot_id]
        x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]
        y = model.generate(x, n, temperature=0.8, top_k=200, eos_token_id=tok.eot_id)
        model.train()
        return tok.decode(y[0].tolist())

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val = float("inf")
    t0 = time.time()
    for it in range(args.max_iters + 1):
        if it % args.eval_interval == 0:
            losses = estimate_loss()
            ppl = math.exp(losses["val"])
            print(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                  f"| val ppl {ppl:6.2f} | {time.time() - t0:5.0f}s")
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({
                    "model": model.state_dict(),
                    "config": asdict(cfg),  # plain dict -> loadable with weights_only=True
                    "vocab_size": vocab_size,
                    "tokenizer_path": TOKENIZER_PATH,
                    "iter": it,
                    "val_loss": losses["val"],
                }, CKPT_PATH)
                print(f"  -> saved {os.path.relpath(CKPT_PATH)} "
                      f"(val {losses['val']:.4f}, ppl {ppl:.2f})")

        if it > 0 and it % args.sample_interval == 0:
            text = sample().replace("\n", " ")
            print(f"  sample: {text[:300]}{' ...' if len(text) > 300 else ''}")

        if it == args.max_iters:
            break

        x, y = get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        optimizer.step()

    print(f"done. best val loss {best_val:.4f} (ppl {math.exp(best_val):.2f})")


if __name__ == "__main__":
    main()

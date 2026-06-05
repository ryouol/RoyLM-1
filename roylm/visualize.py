"""Diagnostic plots for a trained RoyLM-1 checkpoint (token-level versions of
RoyLM-0's plots).

    python -m roylm.visualize

Writes three PNGs to the repo root:
  viz_nexttoken.png   P(next token) after a prompt
  viz_attention.png   per-head attention for a short prompt (last layer)
  viz_embeddings.png  learned token embeddings projected to 2D (PCA)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from .config import CKPT_PATH, GPTConfig, ROOT, TOKENIZER_PATH  # noqa: E402
from .model import GPT  # noqa: E402
from .tokenizer_bpe import BPETokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Visualize RoyLM-1")
    ap.add_argument("--ckpt", default=CKPT_PATH)
    ap.add_argument("--prompt", default="Once upon a time, there")
    ap.add_argument("--layer", type=int, default=-1)
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        raise SystemExit(f"No checkpoint at {args.ckpt}. Train one first: python -m roylm.train")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    cfg = GPTConfig(**ckpt["config"])
    tok = BPETokenizer.load(ckpt.get("tokenizer_path", TOKENIZER_PATH))

    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ids = tok.encode(args.prompt)
    x = torch.tensor(ids, dtype=torch.long)[None, ...]
    with torch.no_grad():
        logits, _ = model(x)

    # --- next-token probabilities ---
    probs = torch.softmax(logits[0, -1], dim=-1)
    top = torch.topk(probs, 15)
    labels = [repr(tok.id_to_token(i.item())) for i in top.indices]
    plt.figure(figsize=(9, 4))
    plt.bar(range(15), top.values.numpy(), color="#4C72B0")
    plt.xticks(range(15), labels, rotation=45, ha="right")
    plt.ylabel("probability")
    plt.title(f"P(next token) after {args.prompt!r}")
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "viz_nexttoken.png"), dpi=120)
    plt.close()
    print("saved viz_nexttoken.png")

    # --- attention heatmaps (one per head, chosen layer) ---
    att = model.transformer.h[args.layer].attn.last_att[0]  # (n_head, T, T)
    toks = [tok.id_to_token(i) for i in ids]
    nh = att.shape[0]
    fig, axes = plt.subplots(1, nh, figsize=(2.6 * nh, 3.0))
    if nh == 1:
        axes = [axes]
    for h in range(nh):
        ax = axes[h]
        ax.imshow(att[h].numpy(), cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"head {h}", fontsize=9)
        ax.set_xticks(range(len(toks)))
        ax.set_xticklabels(toks, rotation=90, fontsize=6)
        if h == 0:
            ax.set_yticks(range(len(toks)))
            ax.set_yticklabels(toks, fontsize=6)
        else:
            ax.set_yticks([])
    fig.suptitle(f"attention  layer {args.layer}  |  {args.prompt!r}  (causal)")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "viz_attention.png"), dpi=120)
    plt.close(fig)
    print("saved viz_attention.png")

    # --- token embeddings projected to 2D (PCA via SVD) ---
    W = model.transformer.wte.weight.detach().numpy().astype(np.float64)
    with np.errstate(all="ignore"):  # Apple Accelerate emits spurious FP warnings here
        Wc = W - W.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(Wc, full_matrices=False)
        xy = Wc @ Vt[:2].T
    plt.figure(figsize=(9, 9))
    plt.scatter(xy[:, 0], xy[:, 1], s=4, alpha=0.15, color="#C44E52")
    annotated = 0
    for i in range(W.shape[0]):
        clean = (tok.id_to_token(i) or "").replace(" ", "")
        if len(clean) >= 3 and clean.isalpha() and i % 5 == 0:
            plt.annotate(clean, (xy[i, 0], xy[i, 1]), fontsize=7, alpha=0.7)
            annotated += 1
        if annotated >= 45:
            break
    plt.title("RoyLM-1 learned token embeddings (PCA to 2D)")
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "viz_embeddings.png"), dpi=120)
    plt.close()
    print("saved viz_embeddings.png")


if __name__ == "__main__":
    main()

"""Generate text from a trained RoyLM-1 checkpoint.

    python -m roylm.sample --prompt "Once upon a time" --tokens 300
    python -m roylm.sample --temperature 0.7 --top-p 0.9 --num-samples 3

RoyLM-1 continues text in the style of TinyStories. It is a base language
model, not a chatbot -- it does not answer questions.
"""
import argparse

import torch

from .config import CKPT_PATH, GPTConfig, TOKENIZER_PATH
from .model import GPT
from .tokenizer_bpe import BPETokenizer


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    ap = argparse.ArgumentParser(description="Sample from RoyLM-1")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--tokens", type=int, default=300, help="max new tokens")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=200, help="0 disables top-k")
    ap.add_argument("--top-p", type=float, default=None, help="nucleus sampling, e.g. 0.9")
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--ckpt", default=CKPT_PATH)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    device = get_device()
    torch.manual_seed(args.seed)

    # weights_only=True loads only tensors and plain data (no arbitrary
    # unpickling). The config is stored as a plain dict for exactly this reason.
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    cfg = GPTConfig(**ckpt["config"])
    tok = BPETokenizer.load(ckpt.get("tokenizer_path", TOKENIZER_PATH))

    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.to(device)

    top_k = args.top_k if args.top_k and args.top_k > 0 else None
    ids = tok.encode(args.prompt) or [tok.eot_id]
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]

    for i in range(args.num_samples):
        y = model.generate(
            x, args.tokens,
            temperature=args.temperature,
            top_k=top_k,
            top_p=args.top_p,
            eos_token_id=tok.eot_id,
        )
        print("=" * 64)
        print(tok.decode(y[0].tolist()))
    print("=" * 64)


if __name__ == "__main__":
    main()

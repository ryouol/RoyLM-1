"""Interactive prompting for RoyLM-1 -- type text and watch it continue.

    python -m roylm.prompt

RoyLM-1 is a *base* model trained on TinyStories: it CONTINUES your text in
that style, it does not answer questions. Type the start of a story and press
Enter. The model (and tokenizer) load once, so you can iterate quickly.

Slash-commands tweak sampling on the fly:
    /temp 0.7      temperature (higher = more random, lower = more predictable)
    /topk 100      top-k        (0 disables)
    /topp 0.9      top-p/nucleus (off/none disables)
    /tokens 200    how many new tokens to generate
    /settings      show current settings
    /help          show this help
    /quit          exit  (also Ctrl-C / Ctrl-D)
"""
import argparse

import torch

from .config import CKPT_PATH, GPTConfig, TOKENIZER_PATH
from .model import GPT
from .tokenizer_bpe import BPETokenizer

HELP = """commands:
  /temp X     set temperature (e.g. 0.7)
  /topk X     set top-k (0 disables)
  /topp X     set top-p / nucleus (off to disable)
  /tokens X   set new tokens to generate
  /settings   show current settings
  /help       show this help
  /quit       exit"""


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def handle_command(line, settings):
    """Mutate `settings` in place from a /command line. Returns False to quit."""
    parts = line[1:].split()
    cmd = parts[0].lower() if parts else ""
    arg = parts[1] if len(parts) > 1 else None

    if cmd in ("quit", "q", "exit"):
        return False
    if cmd in ("help", "h", "?"):
        print(HELP)
    elif cmd in ("settings", "s"):
        _show_settings(settings)
    elif cmd in ("temp", "temperature") and arg:
        settings["temperature"] = float(arg)
        _show_settings(settings)
    elif cmd in ("topk", "top-k", "k") and arg:
        settings["top_k"] = int(arg)
        _show_settings(settings)
    elif cmd in ("topp", "top-p", "p") and arg:
        settings["top_p"] = None if arg.lower() in ("off", "none", "0") else float(arg)
        _show_settings(settings)
    elif cmd in ("tokens", "n") and arg:
        settings["tokens"] = int(arg)
        _show_settings(settings)
    else:
        print(f"  unknown command: /{cmd}  (try /help)")
    return True


def _show_settings(s):
    print(f"  temperature={s['temperature']}  top_k={s['top_k']}  "
          f"top_p={s['top_p']}  tokens={s['tokens']}")


def main():
    ap = argparse.ArgumentParser(description="Interactive prompting for RoyLM-1")
    ap.add_argument("--ckpt", default=CKPT_PATH)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--tokens", type=int, default=200)
    args = ap.parse_args()

    device = get_device()
    try:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    except FileNotFoundError:
        raise SystemExit(
            f"No checkpoint at {args.ckpt}.\nTrain one first:  python -m roylm.train"
        )

    cfg = GPTConfig(**ckpt["config"])
    tok = BPETokenizer.load(ckpt.get("tokenizer_path", TOKENIZER_PATH))
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.to(device)

    settings = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "tokens": args.tokens,
    }

    print(f"RoyLM-1 ready: {model.num_params() / 1e6:.2f}M params, "
          f"vocab {cfg.vocab_size}, device {device}"
          + (f", from iter {ckpt['iter']}" if "iter" in ckpt else ""))
    print("It CONTINUES text in TinyStories style; it does not answer questions.")
    print('Type a prompt (e.g. "Once upon a time"), or /help. Ctrl-C to quit.\n')

    while True:
        try:
            text = input("you > ")
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break

        if not text.strip():
            continue
        if text.startswith("/"):
            if not handle_command(text, settings):
                print("bye")
                break
            continue

        ids = tok.encode(text) or [tok.eot_id]
        x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]
        top_k = settings["top_k"] if settings["top_k"] and settings["top_k"] > 0 else None
        y = model.generate(
            x, settings["tokens"],
            temperature=settings["temperature"],
            top_k=top_k,
            top_p=settings["top_p"],
            eos_token_id=tok.eot_id,
        )
        print("roylm-1 >", tok.decode(y[0].tolist()), "\n")


if __name__ == "__main__":
    main()

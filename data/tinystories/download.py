"""Download (a slice of) the TinyStories dataset as plain text.

TinyStories is a corpus of short, simple stories generated to be learnable by
very small language models -- which makes it ideal for RoyLM-1. The full
training file is ~2 GB, so by default we stream only the first --max-mb
megabytes of the train split, which is plenty for a laptop-scale model. The
validation split (~20 MB) is downloaded in full.

    python data/tinystories/download.py                 # ~200 MB train slice
    python data/tinystories/download.py --max-mb 0       # full ~2 GB train
    python data/tinystories/download.py --max-mb 5       # tiny slice (quick test)
"""
import argparse
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"

# filename -> default cap in MB (0 means download the whole file)
FILES = {
    "TinyStories-train.txt": 200.0,
    "TinyStories-valid.txt": 0.0,
}

CHUNK = 1 << 20  # 1 MB


def download(name, cap_mb):
    url = f"{BASE}/{name}"
    dest = os.path.join(HERE, name)
    if os.path.exists(dest):
        print(f"{name}: already present ({os.path.getsize(dest) / 1e6:.1f} MB), skipping")
        return

    cap = int(cap_mb * 1e6) if cap_mb and cap_mb > 0 else None
    print(f"{name}: downloading{' first %.0f MB' % cap_mb if cap else ''} ...")
    read = 0
    truncated = False
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as f:
        while True:
            chunk = resp.read(CHUNK)
            if not chunk:
                break
            if cap is not None and read + len(chunk) >= cap:
                f.write(chunk[: cap - read])
                read = cap
                truncated = True
                break
            f.write(chunk)
            read += len(chunk)
            print(f"\r  {read / 1e6:7.1f} MB", end="", flush=True)
    print(f"\r  wrote {read / 1e6:.1f} MB -> {os.path.relpath(dest)}")

    # A byte cap can land mid-line; trim back to the last complete line so we
    # never feed a half-written story into the tokenizer.
    if truncated:
        _trim_to_last_newline(dest)


def _trim_to_last_newline(path):
    with open(path, "rb") as f:
        data = f.read()
    nl = data.rfind(b"\n")
    if nl != -1 and nl != len(data) - 1:
        with open(path, "wb") as f:
            f.write(data[: nl + 1])


def main():
    ap = argparse.ArgumentParser(description="Download TinyStories text")
    ap.add_argument("--max-mb", type=float, default=None,
                    help="cap applied to every file (overrides per-file defaults); 0 = full")
    args = ap.parse_args()

    for name, default_cap in FILES.items():
        cap = args.max_mb if args.max_mb is not None else default_cap
        download(name, cap)
    print("done.")


if __name__ == "__main__":
    main()

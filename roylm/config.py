"""Central configuration for RoyLM-1: file paths, tokenizer, model, and training.

Keeping every hyperparameter and path in one place is the first habit that makes
this feel like a real ML project instead of a loose pile of scripts. `model.py`,
`train.py`, `sample.py`, and the data scripts all read from here.
"""
import os
from dataclasses import dataclass

# --- paths ----------------------------------------------------------------
# HERE = .../RoyLM-1/roylm ; ROOT = .../RoyLM-1
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DATA_DIR = os.path.join(ROOT, "data", "tinystories")
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")

TRAIN_TXT = os.path.join(DATA_DIR, "TinyStories-train.txt")
VAL_TXT = os.path.join(DATA_DIR, "TinyStories-valid.txt")
TOKENIZER_PATH = os.path.join(DATA_DIR, "tokenizer.json")
TRAIN_BIN = os.path.join(DATA_DIR, "train.bin")
VAL_BIN = os.path.join(DATA_DIR, "val.bin")
META_PATH = os.path.join(DATA_DIR, "meta.json")
CKPT_PATH = os.path.join(CHECKPOINT_DIR, "roylm-1.pt")

# --- tokenizer ------------------------------------------------------------
# RoyLM-0 had ~65 characters. RoyLM-1 learns a 4,096-token subword vocabulary:
# big enough to capture common chunks ("the", "ing", " once"), small enough to
# train on a laptop.
VOCAB_SIZE = 4096
EOT_TOKEN = "<|endoftext|>"  # marks the boundary between stories


# --- model ----------------------------------------------------------------
@dataclass
class GPTConfig:
    """Architecture. `vocab_size` is overwritten at train time from the
    actual tokenizer so the embedding table and output head match it exactly."""
    block_size: int = 256
    vocab_size: int = VOCAB_SIZE
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.1


# --- training -------------------------------------------------------------
@dataclass
class TrainConfig:
    """Training loop settings. Deliberately close to RoyLM-0 (constant LR,
    AdamW, gradient clipping) so the only real changes are the tokenizer,
    the dataset, and the larger scale. LR schedules come in RoyLM-2."""
    batch_size: int = 16
    learning_rate: float = 3e-4
    max_iters: int = 10000
    eval_interval: int = 500
    eval_iters: int = 100
    grad_clip: float = 1.0
    sample_interval: int = 1000   # generate a short sample every N iters
    sample_tokens: int = 200
    seed: int = 1337

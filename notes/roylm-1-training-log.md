# RoyLM-1 training log

A running record of training runs: configuration, data size, results, and
observations. Validation perplexity (`e^val_loss`) is the headline number to
watch -- lower is better. As a reference point, a uniform random guess over a
4,096-token vocabulary has perplexity 4,096.

## Template

```
### Run YYYY-MM-DD
- Data:        <train MB> train / <val MB> val  (<train tokens> / <val tokens>)
- Tokenizer:   vocab=4096, byte-level BPE
- Model:       block=256, n_embd=256, n_layer=6, n_head=8, dropout=0.1  (~X.XM params)
- Training:    batch=16, lr=3e-4, max_iters=10000, device=mps
- Wall clock:  ~XX min
- Best val:    loss X.XXX  /  perplexity XX.X
- Sample (prompt "Once upon a time", temp 0.8, top_k 200):
    <paste a generated sample>
- Notes:       <what worked, what to try next>
```

## Runs

_No full training runs recorded yet._

The pipeline has been smoke-tested end to end on a tiny slice (a few MB of text,
a small model, a handful of iterations) only to confirm every stage runs and the
shapes line up. Record the first real run above once it finishes.

## Ideas to try

- Longer training (more `max_iters`) and watch when val perplexity stops falling.
- Larger tokenizer vocab (8,192) -- shorter sequences, bigger output layer.
- More data (`download.py --max-mb 0` for the full train split).
- Sampling settings: compare `--temperature`, `--top-k`, and `--top-p 0.9`.

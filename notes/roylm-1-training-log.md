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

### Run 2026-06-04 - starter run
- Data:        200 MB train / 19.4 MB val  (50,180,187 / 4,857,633 tokens; 218,996 / 21,990 stories)
- Tokenizer:   vocab=4096, byte-level BPE
- Model:       block=256, n_embd=256, n_layer=6, n_head=8, dropout=0.1  (5.85M params)
- Training:    batch=16, lr=3e-4, max_iters=2000, device=mps
- Wall clock:  ~3.9 min (234 s)
- Tokens seen: ~8.2M  (~0.16 epochs over the train split)
- Best val:    loss 2.809  /  perplexity 16.59   (random init was ppl 4168.79)
- Val ppl curve: 4168.79 (0) -> 63.12 (250) -> 26.77 (1000) -> 20.15 (1500) -> 16.59 (2000)
- Sample (prompt "Tom and his sister found a box.", temp 0.7, top_k 100):
    "Tom and his sister found a box. They had a toy to make them feel better.
     'Let's go!' Tom said. 'Do you need the car?' They decided to go to the table
     and opened their door. They saw a big box of apples and flour..."
- Notes:       Coherent TinyStories prose after only ~0.16 epochs. Val loss was
               still falling steadily at iter 2000, so a longer run (10k+) should
               keep improving. This was a quick starter; not tuned.

## Ideas to try

- Longer training (more `max_iters`) and watch when val perplexity stops falling.
- Larger tokenizer vocab (8,192) -- shorter sequences, bigger output layer.
- More data (`download.py --max-mb 0` for the full train split).
- Sampling settings: compare `--temperature`, `--top-k`, and `--top-p 0.9`.

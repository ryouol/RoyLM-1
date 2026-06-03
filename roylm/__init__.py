"""RoyLM-1: a small token-level (byte-level BPE) decoder-only GPT.

The transformer is the same architecture as RoyLM-0; the difference is that
RoyLM-1 predicts subword *tokens* from a trained BPE vocabulary instead of
single characters.
"""

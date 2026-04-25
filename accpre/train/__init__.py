"""Offline predictor training (Stage 2A / 2B).

One `Dataset` class per target kind; one `losses.py` module for all
supported loss functions; one `cli.py` trainer that reads a yaml
config. No per-predictor training scripts — the whole matrix is driven
by `accpre/train/cli.py <config.yaml>`.
"""

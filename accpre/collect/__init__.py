"""Stage 2A.0 collection: feature extraction and Stage-1 dataset builder.

All predictor training datasets in Phase 2 come from
`data_collected/stage1.pt`, produced by `accpre/collect/cli.py`. The
file is a list of `RoundRecord`s with the optional feature fields
populated (verifier_hidden, drafter_hidden, prefix_tail, and the three
numeric verifier statistics).
"""

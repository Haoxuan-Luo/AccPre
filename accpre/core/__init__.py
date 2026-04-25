"""Canonical decode mechanics: protocol, schema, accept, commit, draft_verify.

This package is the single source of truth for speculative diffusion decoding
logic in v1. Every other package imports from here; nothing in here imports
from `predictors/`, `train/`, `eval/`, or `reference/`.
"""

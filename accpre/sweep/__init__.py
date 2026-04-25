"""Predictor sweep framework.

Thin orchestration layer around the existing trainer
(`accpre.train.cli`) and the trusted Phase 26 online eval helpers
(`scripts/phase26_conf_sweep.py`).

It does NOT redefine predictor math or evaluation semantics; it only
organises:

  - base configs for the four supported predictor families,
  - a dataset registry (owt wired; others stubbed),
  - a single-run driver that trains + evaluates one config,
  - a grid driver that launches multiple single-run invocations,
  - a summariser that turns per-run metrics into a leaderboard.

See `scripts/sweep_train_eval.py`, `scripts/sweep_grid.py`,
`scripts/sweep_summarize.py` for entry points.
"""

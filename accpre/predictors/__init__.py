"""Predictor method families (Stage 2A/2B).

All predictors inherit from `PredictorBase`. Each declares:
  - `kind`:        "acceptance" or "committed_length"
  - `family`:      which input feature family (see collect/features.py)
  - `deploy_mode`: "V-free" or "V-prefix" (see DESIGN_PHASE2.md §C.3)

The eval harness (`accpre/eval/online_decode.py`, Stage 2A.3)
dispatches on these attributes. No predictor may silently change its
deploy mode or feature family at inference time.
"""

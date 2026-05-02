"""Self-contained vendored copy of the speculative-decoding core.

Mirrors only the modules the coursework experiment touches:

  protocol.py        ProtocolConfig, derive_seed
  commit.py          commit_strict, commit_threshold, commit_confidence
  accept.py          per_position_accept, AcceptOutcome
  draft_verify.py    _make_generator, draft_verify_accept, sample_fallback_or_bonus
  verifier_gpt2.py   GPT2Verifier
  drafter_mdlm.py    MDLMDrafter

Vendored verbatim from `accpre/core/*` and `accpre/models/*` in the parent
project so the homework folder is a stand-alone code submission.
"""

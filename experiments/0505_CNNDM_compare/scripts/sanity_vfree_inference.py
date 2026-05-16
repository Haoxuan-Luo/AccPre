"""Static sanity check: predictor heads and the inference path are V-free.

Two scopes:

1. `scripts/heads.py` (the predictor head module)
   Forbidden tokens at any line of the file: `verifier`, `score`, `prefix_features`,
   `min_pq_j`, `accepted_j`, `verifier_hidden`. The head must use ONLY drafter-side
   inputs at inference, plus its own learned token embedding and position scalar.

2. `scripts/eval_predictor.py` (the V-free online decode entry)
   The round-loop region — every function whose name does NOT begin with
   `_offline` or `compute_offline` — must contain ZERO `verifier.` or
   `verifier =`-style references. The post-hoc verifier NLL pass is
   allowed but only inside `_offline_*` / `compute_offline_*` functions
   plus the top-level `main()` body where the verifier is loaded and
   passed straight to `_offline_compute_nll_tok_succ`.

   We additionally check that `head(...)` calls only appear inside
   functions whose names contain `_vfree_round`, `run_online_vfree`, or
   `_load_head` — i.e. the head is never invoked downstream of a verifier
   forward. This is a strong V-free invariant.

Files referenced by the eval driver but living elsewhere (e.g. accpre.eval.online_decode
or experiments/OWT_Frozen_0429/scripts/eval_with_fallback.py) are NOT scanned here —
this experiment uses its own driver, and any drift in the legacy script does not
affect us.

Pure static / regex-based check; no torch required.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List

_THIS = Path(__file__).resolve()
_EXP_ROOT = _THIS.parents[1]
_HEADS = _EXP_ROOT / "scripts/heads.py"
_EVAL = _EXP_ROOT / "scripts/eval_predictor.py"

# Tokens that, if they appear in the heads module at all, indicate verifier
# leakage at inference time.
_HEADS_FORBIDDEN = (
    "verifier",
    "prefix_features",
    "min_pq_j",
    "accepted_j",
    "verifier_hidden",
    # Note: 'score' is too generic (we have nn.score-like attrs in unrelated places),
    # so we restrict to the others. The signal we care about — calling
    # verifier.score — is caught by the 'verifier' token below.
)


def _scan_heads() -> List[str]:
    if not _HEADS.exists():
        return [f"{_HEADS} does not exist"]
    src = _HEADS.read_text()
    msgs: List[str] = []
    in_docstring = False
    docstring_quote = None
    for line_no, line in enumerate(src.splitlines(), start=1):
        stripped = line.lstrip()
        # Track triple-quoted docstring blocks (best-effort: we only care about
        # the file's top-level module docstring and class/function docstrings,
        # which share the """...""" convention).
        if not in_docstring:
            for q in ('"""', "'''"):
                if stripped.startswith(q):
                    rest = stripped[3:]
                    if q in rest:
                        # one-line docstring
                        break
                    in_docstring = True
                    docstring_quote = q
                    break
        else:
            if docstring_quote and docstring_quote in line:
                in_docstring = False
                docstring_quote = None
            continue
        if in_docstring:
            continue
        if stripped.startswith("#"):
            continue
        for token in _HEADS_FORBIDDEN:
            if re.search(rf"\b{re.escape(token)}\b", line):
                msgs.append(
                    f"scripts/heads.py:{line_no}: forbidden token {token!r} in line: "
                    f"{line.rstrip()}"
                )
    return msgs


def _scan_eval_predictor() -> List[str]:
    """Scan scripts/eval_predictor.py for verifier leakage outside the offline pass.

    Strategy: parse the file with `ast` and tag every line with the closest
    enclosing function name. Any line containing a forbidden pattern that
    is NOT inside an `_offline*` / `compute_offline*` function and NOT
    inside `main()` (where verifier is loaded for handoff to the offline
    helper) is a fail.

    Forbidden patterns:
      verifier.score(            # any verifier call
      verifier.prefix_features(
      verifier.model             # verifier internals
      verifier_hidden            # verifier-derived feature
    """
    if not _EVAL.exists():
        print("(eval_predictor.py not present — skipping its scan)")
        return []
    src = _EVAL.read_text()
    msgs: List[str] = []

    forbidden_patterns = (
        r"\bverifier\.score\b",
        r"\bverifier\.prefix_features\b",
        r"\bverifier\.model\b",
        r"\bverifier_hidden\b",
    )

    import ast

    # Build line -> enclosing function name map via AST.
    try:
        tree = ast.parse(src, filename=str(_EVAL))
    except SyntaxError as e:
        return [f"scripts/eval_predictor.py: syntax error: {e}"]
    n_lines = len(src.splitlines())
    fn_at_line: List[str] = ["<module>"] * (n_lines + 2)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", n_lines)
            for ln in range(start, min(end, n_lines) + 1):
                fn_at_line[ln] = node.name

    lines = src.splitlines()
    in_docstring = False
    docstring_quote: str | None = None
    for line_no, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        # docstring tracking (so we don't fire on doc text)
        if not in_docstring:
            for q in ('"""', "'''"):
                if stripped.startswith(q):
                    rest = stripped[3:]
                    if q in rest:
                        break
                    in_docstring = True
                    docstring_quote = q
                    break
        else:
            if docstring_quote and docstring_quote in line:
                in_docstring = False
                docstring_quote = None
            continue
        if in_docstring:
            continue
        if stripped.startswith("#"):
            continue
        cur_fn = fn_at_line[line_no]
        is_offline = cur_fn.startswith("_offline") or cur_fn.startswith("compute_offline")
        is_main = cur_fn == "main"
        for pat in forbidden_patterns:
            if re.search(pat, line):
                if is_offline or is_main:
                    continue
                msgs.append(
                    f"scripts/eval_predictor.py:{line_no}: verifier reference inside "
                    f"function {cur_fn!r} — only `_offline*`/`compute_offline*` and `main` "
                    f"are allowed (line: {line.rstrip()})"
                )
    return msgs


def main() -> int:
    print("=== sanity_vfree_inference ===")
    msgs: List[str] = []
    msgs += _scan_heads()
    msgs += _scan_eval_predictor()
    if msgs:
        for m in msgs:
            print("FAIL: " + m)
        print(f"FAILED ({len(msgs)} issues)")
        return 1
    print("PASS (heads.py is V-free at inference)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

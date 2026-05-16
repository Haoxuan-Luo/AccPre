"""Static sanity check: this experiment never references `relative_max` as a
commit rule.

Scope:
  - All YAMLs under `experiments/0505_OWT_compare/configs/`.
  - All Python files under `experiments/0505_OWT_compare/scripts/`.
  - All SLURM/shell scripts under `experiments/0505_OWT_compare/jobs/`.

We forbid the literal string "relative_max" anywhere except in comments, and
in this script's own source (which is the one place it must be named to
search for it).

Pure regex check; no torch required.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_EXP_ROOT = _THIS.parents[1]


_SELF_BASENAME = _THIS.name   # "sanity_no_relative_max.py"


def _scan_dir(directory: Path, suffixes: tuple[str, ...]) -> list[str]:
    msgs: list[str] = []
    if not directory.exists():
        return msgs
    for fp in sorted(directory.rglob("*")):
        if not fp.is_file():
            continue
        if fp.suffix not in suffixes:
            continue
        if fp.resolve() == _THIS:
            continue   # this file is allowed to mention the name
        try:
            src = fp.read_text()
        except Exception:
            continue
        in_docstring = False
        docstring_quote = None
        for line_no, line in enumerate(src.splitlines(), start=1):
            stripped = line.lstrip()
            # Track triple-quoted docstring blocks (best-effort).
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
            if in_docstring:
                continue
            if stripped.startswith("#"):
                continue
            # Skip lines that reference our own script filename — this is the one
            # place the literal token must appear (the path to the sanity script).
            if _SELF_BASENAME in line:
                continue
            if "relative_max" in line:
                msgs.append(
                    f"{fp.relative_to(_EXP_ROOT)}:{line_no}: forbidden token "
                    f"'relative_max' in line: {line.rstrip()}"
                )
    return msgs


def main() -> int:
    print("=== sanity_no_relative_max ===")
    msgs: list[str] = []
    msgs += _scan_dir(_EXP_ROOT / "configs", suffixes=(".yaml", ".yml"))
    msgs += _scan_dir(_EXP_ROOT / "scripts", suffixes=(".py",))
    msgs += _scan_dir(_EXP_ROOT / "jobs", suffixes=(".slurm", ".sh"))
    if msgs:
        for m in msgs:
            print("FAIL: " + m)
        print(f"FAILED ({len(msgs)} occurrences)")
        return 1
    print("PASS (no `relative_max` in active configs/scripts/jobs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

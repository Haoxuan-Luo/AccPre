"""Static sanity check: joint configs never load a fixed-target file.

Two checks:
  1. Every YAML under `configs/joint_*.yaml` and `configs/_smoke/joint_*.yaml`:
     - has `mode: joint`
     - has NO `dep_targets_path` (or it is null)
     - has exactly one of {live_relmax_target, live_q2_target, live_dep_target}
       set to true; the other two are absent or false
     - the live-flag matches the file's target (joint_relmax_*.yaml -> live_relmax_target, etc.)
  2. The joint trainer source (`scripts/train_joint.py`) does NOT contain any
     reference to `_T1_relmax.pt`, `_T1_q2.pt`, or `_T1_dep.pt` outside an
     assert / docstring context.

Also checks that the joint training source explicitly passes
`dep_targets_path=None` to JointAcceptanceDataset.

Pure static / regex-based check; no torch required.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Tuple

import yaml

_THIS = Path(__file__).resolve()
_EXP_ROOT = _THIS.parents[1]      # experiments/0505_OWT_compare/
_TRAIN_JOINT = _EXP_ROOT / "scripts/train_joint.py"


_TARGET_TO_FLAG = {
    "relmax": "live_relmax_target",
    "alpha_q2": "live_q2_target",
    "dep": "live_dep_target",
}


def _check_yaml(path: Path) -> Tuple[bool, list[str]]:
    msgs: list[str] = []
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        return False, [f"{path}: not a dict"]

    if cfg.get("mode") != "joint":
        msgs.append(f"{path}: mode != 'joint' (got {cfg.get('mode')!r})")
    if cfg.get("dep_targets_path") is not None:
        msgs.append(
            f"{path}: dep_targets_path must be absent/null in joint configs "
            f"(got {cfg.get('dep_targets_path')!r})"
        )
    target = cfg.get("target")
    if target not in _TARGET_TO_FLAG:
        msgs.append(f"{path}: target {target!r} not in {sorted(_TARGET_TO_FLAG)}")
    else:
        expected_flag = _TARGET_TO_FLAG[target]
        if not bool(cfg.get(expected_flag, False)):
            msgs.append(
                f"{path}: target={target!r} requires `{expected_flag}: true` "
                f"(got {cfg.get(expected_flag)!r})"
            )
        for other in set(_TARGET_TO_FLAG.values()) - {expected_flag}:
            if bool(cfg.get(other, False)):
                msgs.append(
                    f"{path}: target={target!r} but `{other}: true` is also set"
                )
    return (not msgs), msgs


def _scan_yaml_dir(directory: Path, prefix: str) -> Tuple[int, list[str]]:
    """Scan directory for YAMLs whose name begins with prefix; return (n_checked, errors)."""
    if not directory.exists():
        return 0, []
    errors: list[str] = []
    n = 0
    for fp in sorted(directory.glob(f"{prefix}*.yaml")):
        n += 1
        ok, msgs = _check_yaml(fp)
        if not ok:
            errors.extend(msgs)
        else:
            print(f"[ok ] {fp.relative_to(_EXP_ROOT)}")
    return n, errors


def _scan_template(path: Path) -> Tuple[int, list[str]]:
    """Templates do not match the joint_*.yaml prefix but are real config sources;
    we check them too. The template's `live_*_target` flag pattern must be
    consistent with target."""
    if not path.exists():
        return 0, []
    errors: list[str] = []
    ok, msgs = _check_yaml(path)
    if not ok:
        errors.extend(msgs)
    else:
        print(f"[ok ] {path.relative_to(_EXP_ROOT)}  (joint template)")
    return 1, errors


def _check_train_joint_source() -> list[str]:
    """Static grep on the joint trainer source."""
    if not _TRAIN_JOINT.exists():
        return [f"{_TRAIN_JOINT} does not exist"]
    src = _TRAIN_JOINT.read_text()

    msgs: list[str] = []

    # We DO want at least one explicit `dep_targets_path=None` call in JointAcceptanceDataset.
    if not re.search(r"JointAcceptanceDataset\([^)]*dep_targets_path=None", src, re.DOTALL):
        msgs.append(
            "scripts/train_joint.py: must explicitly pass `dep_targets_path=None` "
            "when constructing JointAcceptanceDataset"
        )

    # Forbid any reference to fixed-target file basenames outside assert / docstring.
    forbidden = ("_T1_relmax.pt", "_T1_q2.pt", "_T1_dep.pt")
    for needle in forbidden:
        # Find every occurrence; allow occurrences inside docstrings/comments.
        for m in re.finditer(re.escape(needle), src):
            line_start = src.rfind("\n", 0, m.start()) + 1
            line_end = src.find("\n", m.end())
            if line_end < 0:
                line_end = len(src)
            line = src[line_start:line_end]
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            # Permit occurrence inside a triple-quoted docstring? Detection is hard
            # statically; we conservatively reject any non-comment occurrence.
            msgs.append(
                f"scripts/train_joint.py: forbidden reference to {needle!r} "
                f"(line: {line.rstrip()})"
            )

    return msgs


def main() -> int:
    print("=== sanity_no_fixed_targets_in_joint ===")
    n_total = 0
    all_errs: list[str] = []
    for sub in ("configs", "configs/_smoke"):
        d = _EXP_ROOT / sub
        n, errs = _scan_yaml_dir(d, prefix="joint_")
        n_total += n
        all_errs.extend(errs)
    # Also check the joint template (it materialises into joint_*.yaml at smoke time).
    n_t, errs_t = _scan_template(_EXP_ROOT / "configs/_smoke/_template_joint.yaml")
    n_total += n_t
    all_errs.extend(errs_t)
    src_errs = _check_train_joint_source()
    all_errs.extend(src_errs)

    if all_errs:
        for e in all_errs:
            print("FAIL: " + e)
        print(f"FAILED ({len(all_errs)} errors across {n_total} configs)")
        return 1
    print(f"PASS ({n_total} joint configs OK; train_joint.py OK)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

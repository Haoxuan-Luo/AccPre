"""CNN/DM 300 preflight — run inside the apptainer SIF before any sbatch.

Catches the failure modes that caused a 6-second cnndm300_collect FAIL on the
collaborator's checkout (most likely: a stale clone missing the `cnn_dm_300`
registration in `accpre/`, an unreadable container, an HF-cache permissions
problem, or the eval/train sweep grids drifting out of sync with the 144/96/180
cell counts).

Usage
-----
    apptainer exec --nv <SIF> python3 \\
        experiments/0505_CNNDM_compare/scripts/preflight_cnndm_300.py

Exit code: 0 if every check PASSes, 1 otherwise. Output is one line per check.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, List, Tuple


_THIS = Path(__file__).resolve()
_EXP_ROOT = _THIS.parents[1]
_REPO_ROOT = _THIS.parents[3]

# Project-local Python deps. Must match the value used by setup_collab_env.sh
# and by the APPTAINERENV_PYTHONPATH set in every cnndm_300 slurm file.
_CNNDM_PYDEPS = _EXP_ROOT / ".pydeps"

# Packages that MUST resolve to _CNNDM_PYDEPS (not to ~/.local, not to the
# container's site-packages). If any resolves elsewhere, the batch jobs
# will fail the same way the collaborator's cnndm300_collect did.
_PYDEPS_REQUIRED = ("transformers", "datasets", "dill", "multiprocess")

# Packages that come from the container itself (and should NOT be in .pydeps).
_CONTAINER_REQUIRED = ("torch", "yaml")


# ------------------------------------------------------------------ small
# Result registry

_RESULTS: List[Tuple[str, bool, str]] = []


def _check(name: str, fn: Callable[[], Tuple[bool, str]]) -> None:
    try:
        ok, detail = fn()
    except Exception as e:                         # noqa: BLE001
        ok = False
        detail = f"{type(e).__name__}: {e}"
    _RESULTS.append((name, bool(ok), detail))


# ------------------------------------------------------------------ checks


def _check_repo_root() -> Tuple[bool, str]:
    """REPO_ROOT must contain accpre/, configs/, experiments/."""
    missing = [d for d in ("accpre", "configs", "experiments")
               if not (_REPO_ROOT / d).is_dir()]
    if missing:
        return False, f"missing directories under REPO_ROOT={_REPO_ROOT}: {missing}"
    return True, f"REPO_ROOT={_REPO_ROOT}"


def _grep_for(path: Path, needle: str) -> bool:
    if not path.exists():
        return False
    try:
        return needle in path.read_text()
    except Exception:                              # noqa: BLE001
        return False


def _check_accpre_splits_registered() -> Tuple[bool, str]:
    p = _REPO_ROOT / "accpre/data/splits.py"
    return (_grep_for(p, '"cnn_dm_300"'),
            f"cnn_dm_300 in {p}")


def _check_accpre_prompts_registered() -> Tuple[bool, str]:
    p = _REPO_ROOT / "accpre/data/prompts.py"
    return (_grep_for(p, '"cnn_dm_300"'),
            f"cnn_dm_300 in {p}")


def _check_accpre_collect_cli_registered() -> Tuple[bool, str]:
    p = _REPO_ROOT / "accpre/collect/cli.py"
    return (_grep_for(p, '"cnn_dm_300"'),
            f"cnn_dm_300 in {p}")


def _check_accpre_sweep_datasets_registered() -> Tuple[bool, str]:
    p = _REPO_ROOT / "accpre/sweep/datasets.py"
    return (_grep_for(p, '"cnn_dm_300"'),
            f"cnn_dm_300 in {p}")


def _check_split_config() -> Tuple[bool, str]:
    """get_split_config('cnn_dm_300') returns the expected 300/160/40/100 layout."""
    sys.path.insert(0, str(_REPO_ROOT))
    from accpre.data.splits import get_split_config
    cfg = get_split_config("cnn_dm_300")
    expected = dict(pool_size=300, train_n=160, val_n=40, test_n=100, prefix_len=32)
    got = dict(pool_size=cfg.pool_size, train_n=cfg.train_n,
               val_n=cfg.val_n, test_n=cfg.test_n, prefix_len=cfg.prefix_len)
    if got != expected:
        return False, f"got {got}, expected {expected}"
    return True, f"pool=300 train=160 val=40 test=100 prefix_len=32"


def _check_protocol_yaml() -> Tuple[bool, str]:
    """configs/protocol.yaml exists and pins temperature=1.0."""
    import yaml
    p = _REPO_ROOT / "configs/protocol.yaml"
    if not p.exists():
        return False, f"{p} missing"
    cfg = yaml.safe_load(p.read_text())
    temp = cfg.get("temperature")
    if temp != 1.0:
        return False, f"temperature={temp!r}, expected 1.0"
    return True, "temperature=1.0"


_CNNDM_300_SLURMS = (
    "job_collect_cnndm_300.slurm",
    "job_build_targets_cnndm_300.slurm",
    "job_baselines_cnndm_300.slurm",
    "job_train_frozen_cnndm_300.slurm",
    "job_train_joint_cnndm_300.slurm",
    "job_pick_best_cnndm_300.slurm",
    "job_eval_predictors_cnndm_300.slurm",
    "job_aggregate_cnndm_300.slurm",
)


def _check_all_8_slurm_files_exist() -> Tuple[bool, str]:
    jobs_dir = _EXP_ROOT / "jobs"
    missing = [n for n in _CNNDM_300_SLURMS
               if not (jobs_dir / n).is_file()]
    if missing:
        return False, f"missing: {missing}"
    return True, f"all 8 present in {jobs_dir}"


def _check_no_hardcoded_account() -> Tuple[bool, str]:
    """No '#SBATCH -A' directive in any cnndm_300 slurm file."""
    offenders: List[str] = []
    for n in _CNNDM_300_SLURMS:
        p = _EXP_ROOT / "jobs" / n
        try:
            for line in p.read_text().splitlines():
                if line.strip().startswith("#SBATCH -A"):
                    offenders.append(f"{n}:{line.strip()}")
                    break
        except Exception:                          # noqa: BLE001
            offenders.append(f"{n}:READ_ERROR")
    if offenders:
        return False, f"offenders: {offenders}"
    return True, "no #SBATCH -A in any cnndm_300 slurm"


def _check_no_owt_input_paths_in_full_run() -> Tuple[bool, str]:
    """The cnndm_300 slurm files and the submit script must not READ from
    experiments/0505_OWT_compare/ or experiments/OWT_Frozen_0429/results/.
    Documentation comments referring to those paths are allowed."""
    bad_tokens = ("experiments/0505_OWT_compare/results",
                  "experiments/OWT_Frozen_0429/results")
    files = [_EXP_ROOT / "jobs" / n for n in _CNNDM_300_SLURMS]
    files.append(_EXP_ROOT / "submit_full_cnndm_300.sh")
    offenders: List[str] = []
    for p in files:
        if not p.exists():
            continue
        for i, line in enumerate(p.read_text().splitlines(), start=1):
            if any(t in line for t in bad_tokens):
                offenders.append(f"{p.name}:{i}")
    if offenders:
        return False, f"offenders: {offenders}"
    return True, "no OWT result/baseline input paths in cnndm_300 chain"


def _check_relmax_invariant_via_sanity_script() -> Tuple[bool, str]:
    """Delegate to sanity_no_relative_max.py which already handles
    comment/docstring skipping correctly. The function name and all messages
    in this function intentionally reference sanity_no_relative_max.py (with
    `.py` suffix) so the canonical scanner's self-allowlist rule applies."""
    sanity = _EXP_ROOT / "scripts" / "sanity_no_relative_max.py"
    if not sanity.exists():
        return False, f"sanity_no_relative_max.py not found at {sanity}"
    r = subprocess.run(
        [sys.executable, str(sanity)],
        cwd=str(_REPO_ROOT),
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        tail = (r.stdout + r.stderr).strip().splitlines()[-5:]
        return False, "sanity_no_relative_max.py FAIL: " + " | ".join(tail)
    return True, "sanity_no_relative_max.py PASS"


def _check_apptainer_sif() -> Tuple[bool, str]:
    sif = Path("/share/resources/containers/apptainer/pytorch-2.7.0.sif")
    if not sif.exists():
        return False, f"{sif} missing"
    if not os.access(sif, os.R_OK):
        return False, f"{sif} present but not readable by this UID"
    sz = sif.stat().st_size
    return True, f"{sif} readable ({sz} bytes)"


def _check_cnndm_pydeps_dir() -> Tuple[bool, str]:
    """The project-local CNNDM_PYDEPS dir must exist and contain package
    DIRECTORIES (not just dist-info markers) for every required package."""
    if not _CNNDM_PYDEPS.is_dir():
        return False, (
            f"{_CNNDM_PYDEPS} not present. Run ONCE on a login node: "
            "`bash experiments/0505_CNNDM_compare/scripts/setup_collab_env.sh`"
        )
    missing = [r for r in _PYDEPS_REQUIRED
               if not (_CNNDM_PYDEPS / r).is_dir()]
    if missing:
        return False, (
            f"{_CNNDM_PYDEPS} missing required package dirs: {missing}. "
            "Re-run setup_collab_env.sh to repopulate."
        )
    return True, f"{_CNNDM_PYDEPS} contains {', '.join(_PYDEPS_REQUIRED)}"


def _check_user_site_disabled() -> Tuple[bool, str]:
    """sys.flags.no_user_site must be True in the preflight invocation.

    This proves the caller (submit_full_cnndm_300.sh or the batch slurm
    preflight job) exported PYTHONNOUSERSITE=1, matching what every other
    cnndm_300 slurm file does. If a future change accidentally drops that
    export, this check catches it BEFORE any sbatch is fired."""
    if not sys.flags.no_user_site:
        return False, (
            "PYTHONNOUSERSITE was NOT set in this preflight invocation — "
            "this preflight is running with ~/.local visible, which means "
            "it does NOT mirror the batch environment. Re-run via "
            "`bash experiments/0505_CNNDM_compare/submit_full_cnndm_300.sh "
            "--preflight-only` so APPTAINERENV_PYTHONNOUSERSITE=1 is set."
        )
    return True, "PYTHONNOUSERSITE=1 (~/.local invisible to imports)"


def _check_pydeps_in_sys_path() -> Tuple[bool, str]:
    """_CNNDM_PYDEPS must appear in sys.path. This proves the caller
    exported PYTHONPATH to include the same dependency dir batch jobs use."""
    target = str(_CNNDM_PYDEPS)
    if target not in sys.path:
        return False, (
            f"{target} not in sys.path. The caller must export "
            f"APPTAINERENV_PYTHONPATH=<repo_root>:{target} before "
            "`apptainer exec ... python3 preflight_cnndm_300.py`."
        )
    return True, f"{target} on sys.path"


def _check_python_imports() -> Tuple[bool, str]:
    """Import every module the pipeline depends on at runtime.

    PRECONDITIONS (verified by other checks): PYTHONNOUSERSITE=1 +
    PYTHONPATH includes _CNNDM_PYDEPS. So these imports must resolve via
    either the container's site-packages (torch, yaml) or the project-local
    .pydeps (transformers, datasets, dill, multiprocess). The accpre.*
    imports are pure-Python source under _REPO_ROOT and resolve via
    _REPO_ROOT being on sys.path.
    """
    sys.path.insert(0, str(_REPO_ROOT))
    needed = (
        *_CONTAINER_REQUIRED,
        *_PYDEPS_REQUIRED,
        "accpre.collect.cli",
        "accpre.data.prompts",
        "accpre.data.splits",
        "accpre.sweep.datasets",
        "accpre.core.protocol",
        "accpre.core.schema",
    )
    failed: List[str] = []
    for mod in needed:
        try:
            __import__(mod)
        except Exception as e:                     # noqa: BLE001
            failed.append(f"{mod}({type(e).__name__})")
    if failed:
        hint = ""
        pydeps_missing = set(_PYDEPS_REQUIRED) & {
            f.split("(", 1)[0] for f in failed
        }
        if pydeps_missing:
            hint = (
                "  -- LIKELY CAUSE: .pydeps not populated for these packages. "
                "Run ONCE on a login node: "
                "`bash experiments/0505_CNNDM_compare/scripts/setup_collab_env.sh`."
            )
        return False, f"failed: {failed}{hint}"
    return True, f"all {len(needed)} imports OK"


def _check_pydeps_resolution() -> Tuple[bool, str]:
    """The 4 .pydeps-required packages must resolve to a path UNDER
    _CNNDM_PYDEPS. If any resolves to ~/.local or the container, the same
    package will be missing or differ on a batch node where ~/.local is
    invisible — which is the exact failure mode this whole change fixes."""
    bad: List[str] = []
    sys.path.insert(0, str(_REPO_ROOT))
    pydeps_abs = str(_CNNDM_PYDEPS.resolve())
    for mod in _PYDEPS_REQUIRED:
        m = __import__(mod)
        f = getattr(m, "__file__", "") or ""
        if not f:
            bad.append(f"{mod}: no __file__")
            continue
        if not str(Path(f).resolve()).startswith(pydeps_abs):
            bad.append(f"{mod} resolved to {f} (expected under {pydeps_abs})")
    if bad:
        return False, "wrong resolution: " + " | ".join(bad)
    return True, f"all {len(_PYDEPS_REQUIRED)} resolve under {pydeps_abs}"


def _check_collect_cli_lists_cnn_dm_300() -> Tuple[bool, str]:
    """`python -m accpre.collect.cli --help` must mention 'cnn_dm_300' in the
    --dataset choices."""
    r = subprocess.run(
        [sys.executable, "-m", "accpre.collect.cli", "--help"],
        cwd=str(_REPO_ROOT),
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        return False, f"--help exited {r.returncode}; stderr head: {r.stderr[:200]!r}"
    if "cnn_dm_300" not in (r.stdout + r.stderr):
        return False, "'cnn_dm_300' not in --help output (stale clone?)"
    return True, "cnn_dm_300 listed as a --dataset choice"


def _check_train_sweep_counts() -> Tuple[bool, str]:
    sys.path.insert(0, str(_EXP_ROOT))
    from scripts.train_sweep import build_cell_list
    frozen = sum(1 for c in build_cell_list(regimes=("frozen",)))
    joint = sum(1 for c in build_cell_list(regimes=("joint",)))
    if (frozen, joint) != (144, 96):
        return False, f"got frozen={frozen} joint={joint}; expected 144/96"
    return True, "frozen=144 joint=96"


def _check_eval_sweep_180_cells() -> Tuple[bool, str]:
    """eval_sweep should expand a 15-entry best_per_cell into 180 cells."""
    sys.path.insert(0, str(_EXP_ROOT))
    # Synthesize a best_per_cell.json with the canonical 15 (regime,target,arch) keys.
    best = {}
    for regime, archs in (
        ("frozen", ("mlp_pos", "causal_transformer_pos", "bidirectional_transformer_pos")),
        ("joint",  ("causal_transformer_pos", "bidirectional_transformer_pos")),
    ):
        for target in ("relmax", "alpha_q2", "dep"):
            for arch in archs:
                key = f"{regime}.{target}.{arch}"
                best[key] = {
                    "regime": regime, "target": target, "arch": arch,
                    "run_dir": f"/tmp/preflight/{regime}/{target}/{arch}",
                    "best_val_weighted_mse": 0.0,
                }
    if len(best) != 15:
        return False, f"unexpected best size: {len(best)}"
    with tempfile.TemporaryDirectory() as td:
        bp = Path(td) / "best.json"
        bp.write_text(json.dumps(best))
        r = subprocess.run(
            [sys.executable, str(_EXP_ROOT / "scripts/eval_sweep.py"),
             "--best_per_cell", str(bp),
             "--eval_root", str(Path(td) / "eval"),
             "--dataset", "cnn_dm_300",
             "--gamma", "15", "--T", "1",
             "--dry_run"],
            cwd=str(_REPO_ROOT),
            capture_output=True, text=True, timeout=60,
        )
    if r.returncode != 0:
        return False, f"eval_sweep --dry_run exited {r.returncode}: {r.stderr[:300]!r}"
    out = r.stdout + r.stderr
    for line in out.splitlines():
        if "total_cells" in line:
            # Expect: "[eval-sweep] total_cells=180 shard=0/1 my_cells=180"
            if "total_cells=180" in line:
                return True, "eval cells = 180"
            return False, f"got: {line.strip()}"
    return False, f"could not find total_cells line in eval_sweep output"


# ------------------------------------------------------------------ main


def main() -> int:
    print("=== CNN/DM 300 preflight ===")
    print(f"  REPO_ROOT={_REPO_ROOT}")
    print(f"  EXP_ROOT={_EXP_ROOT}")
    print(f"  CNNDM_PYDEPS={_CNNDM_PYDEPS}")
    print(f"  python={sys.executable}")
    print(f"  no_user_site={sys.flags.no_user_site}")
    print()

    _check("repo_root",                    _check_repo_root)
    _check("accpre/data/splits.py: cnn_dm_300",
                                            _check_accpre_splits_registered)
    _check("accpre/data/prompts.py: cnn_dm_300",
                                            _check_accpre_prompts_registered)
    _check("accpre/collect/cli.py: cnn_dm_300",
                                            _check_accpre_collect_cli_registered)
    _check("accpre/sweep/datasets.py: cnn_dm_300",
                                            _check_accpre_sweep_datasets_registered)
    _check("split_config(cnn_dm_300) values",
                                            _check_split_config)
    _check("configs/protocol.yaml T=1",     _check_protocol_yaml)
    _check("8 cnndm_300 slurm files exist", _check_all_8_slurm_files_exist)
    _check("no hardcoded #SBATCH -A in cnndm_300 slurm",
                                            _check_no_hardcoded_account)
    _check("no OWT result/baseline INPUT paths in cnndm_300 chain",
                                            _check_no_owt_input_paths_in_full_run)
    # The label below references sanity_no_relative_max.py (with `.py` suffix)
    # so the canonical scanner's self-allowlist applies to this line.
    _check("sanity_no_relative_max.py invariant holds",
                                            _check_relmax_invariant_via_sanity_script)
    _check("apptainer SIF readable",        _check_apptainer_sif)
    # The four checks below verify the project-local .pydeps wiring that
    # replaces the old fragile ~/.local user-site path. They MUST all pass
    # together: if .pydeps lacks a package, or if the caller forgot to set
    # PYTHONNOUSERSITE/PYTHONPATH, batch jobs will fail the same way the
    # collaborator's cnndm300_collect did (6-second ModuleNotFoundError).
    _check(".pydeps dir populated",         _check_cnndm_pydeps_dir)
    _check("PYTHONNOUSERSITE=1 (no ~/.local leak)",
                                            _check_user_site_disabled)
    _check(".pydeps on sys.path",           _check_pydeps_in_sys_path)
    _check("python imports (torch, transformers, datasets, accpre.*)",
                                            _check_python_imports)
    _check(".pydeps packages resolve under .pydeps (not ~/.local)",
                                            _check_pydeps_resolution)
    _check("accpre.collect.cli --help lists cnn_dm_300",
                                            _check_collect_cli_lists_cnn_dm_300)
    _check("train_sweep cell counts (144 frozen + 96 joint)",
                                            _check_train_sweep_counts)
    _check("eval_sweep cell count (180)",   _check_eval_sweep_180_cells)

    n_pass = sum(1 for _, ok, _ in _RESULTS if ok)
    n_fail = sum(1 for _, ok, _ in _RESULTS if not ok)
    width = max(len(name) for name, _, _ in _RESULTS) + 2

    for name, ok, detail in _RESULTS:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name.ljust(width)} {detail}")

    print()
    print(f"=== {n_pass} PASS, {n_fail} FAIL ===")
    if n_fail:
        print("Preflight FAILED. Fix the items above before submitting jobs.")
        return 1
    print("Preflight OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

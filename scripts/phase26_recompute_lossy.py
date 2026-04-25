"""Phase 26 — recompute lossy rnd_mean / all_pass with all-rounds denominator.

One-shot correction of the denominator asymmetry identified in the Phase 26
audit: the lossy lane excluded rounds with L_lossy=0 from rnd_mean / all_pass
denominators, while predictor lanes (which always commit ≥ 1 token) always
count every round. This biased lossy rnd_mean / all_pass upward.

This script:
  1. Corrects each `online_lossy_l_*.json` per-prompt entry in place so that
     the rnd_mean / all_pass values use the all-rounds denominator.
  2. Rebuilds `unified_table.{md,json}` by re-aggregating all lanes from the
     on-disk JSONs (NLL is reused from the existing unified_table because
     the decode trajectories — and therefore the offline NLL — are
     unchanged).
  3. Invokes `scripts/phase26_plots.py` to regenerate the Pareto figures and
     the condensed table from the refreshed unified table.

No decode, no predictor training, no verifier forward. The correction uses
information already in each JSON: per-round `L_lossy`, and the per-prompt
aggregates before correction. Under the all-rounds denominator:

    rnd_mean_new  = rnd_mean_old  *  (n_rounds_nonzero / n_rounds_total)
    all_pass_new  = all_pass_old  *  (n_rounds_nonzero / n_rounds_total)
    tok_succ_new  = tok_succ_old     (unchanged — L_lossy=0 rounds
                                      contribute 0 to both numerator and
                                      denominator of tok_succ)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _correct_lossy_json(path: Path) -> Dict[str, object]:
    """Apply the all-rounds denominator correction in place.

    Returns a per-file summary with before/after aggregates.
    """
    with open(path) as f:
        lane = json.load(f)

    before: Dict[str, float] = {"rnd_mean": 0.0, "all_pass": 0.0, "tok_succ": 0.0}
    after: Dict[str, float] = {"rnd_mean": 0.0, "all_pass": 0.0, "tok_succ": 0.0}

    pp = lane["per_prompt"]
    n_prompts = 0
    for p_row in pp:
        rounds = p_row.get("rounds", [])
        n_total = len(rounds)
        if n_total == 0:
            continue
        n_nonzero = sum(1 for r in rounds if int(r.get("L_lossy", 0)) > 0)
        f_corr = float(n_nonzero) / float(n_total)

        old_rnd = float(p_row.get("rnd_mean", 0.0))
        old_all = float(p_row.get("all_pass", 0.0))
        old_tok = float(p_row.get("tok_succ", 0.0))

        new_rnd = old_rnd * f_corr
        new_all = old_all * f_corr
        # tok_succ is invariant: L_lossy=0 rounds contribute 0/0 to both
        # numerator and denominator.
        new_tok = old_tok

        p_row["rnd_mean"] = new_rnd
        p_row["all_pass"] = new_all
        p_row["tok_succ"] = new_tok
        # Diagnostic fields for traceability.
        p_row["n_rounds_total"] = int(n_total)
        p_row["n_rounds_nonzero"] = int(n_nonzero)

        before["rnd_mean"] += old_rnd
        before["all_pass"] += old_all
        before["tok_succ"] += old_tok
        after["rnd_mean"] += new_rnd
        after["all_pass"] += new_all
        after["tok_succ"] += new_tok
        n_prompts += 1

    denom = max(n_prompts, 1)
    summary = {
        "path": str(path),
        "n_prompts": n_prompts,
        "rnd_mean_before": before["rnd_mean"] / denom,
        "rnd_mean_after":  after["rnd_mean"] / denom,
        "all_pass_before": before["all_pass"] / denom,
        "all_pass_after":  after["all_pass"] / denom,
        "tok_succ_before": before["tok_succ"] / denom,
        "tok_succ_after":  after["tok_succ"] / denom,
    }

    # Flag correction in the lane-level metadata so the JSON records its
    # provenance.
    lane["rnd_mean_denominator"] = "all_rounds"
    lane["all_pass_denominator"] = "all_rounds"

    with open(path, "w") as f:
        json.dump(lane, f, indent=2, default=str)
    return summary


def _filename_for_row(r: Dict) -> str:
    fam = r["family"]
    rule = r["rule"]
    lab = r.get("threshold_label", "")
    if fam == "strict":
        return "online_strict.json"
    if fam == "lossy":
        l_val = float(lab.split("=", 1)[1])
        whole = int(l_val)
        frac = int(round((l_val - whole) * 10))
        return f"online_lossy_l_{whole}p{frac}.json"
    tau = float(lab.split("=", 1)[1])
    tau_tag = f"0p{int(round(tau * 10))}"
    rule_tag = "thresh" if rule == "threshold" else "conf"
    return f"online_{fam}_{rule_tag}_tau_{tau_tag}.json"


def _aggregate_row(lane: Dict, r_existing: Dict) -> Dict:
    """Mirror phase26_conf_sweep._row but read per-prompt fields from disk.

    Reuses NLL / delta_NLL / tok_s_mean from the existing row (those are
    unaffected by the correction).
    """
    pp = lane["per_prompt"]
    tot_n = sum(int(p.get("n_tokens_total", 0)) for p in pp)
    tok_succ = (
        sum(
            float(p.get("tok_succ", 0.0)) * int(p.get("n_tokens_total", 0))
            for p in pp
        ) / max(tot_n, 1)
    ) if tot_n > 0 else None
    pp_with_rnd = [p for p in pp if "rnd_mean" in p]
    rnd_mean = (
        sum(p["rnd_mean"] for p in pp_with_rnd) / len(pp_with_rnd)
    ) if pp_with_rnd else None
    pp_with_ap = [p for p in pp if "all_pass" in p]
    all_pass = (
        sum(p["all_pass"] for p in pp_with_ap) / len(pp_with_ap)
    ) if pp_with_ap else None
    n_rounds_per_prompt = [len(p.get("rounds", [])) for p in pp]
    n_committed_per_round = [
        (int(p.get("n_tokens_total", 0)) / max(len(p.get("rounds", [])), 1))
        for p in pp
    ]
    new_row = dict(r_existing)
    new_row["tok_s_mean"] = float(lane.get("tok_s_mean", r_existing["tok_s_mean"]))
    new_row["tok_succ"] = None if tot_n == 0 else float(tok_succ)
    new_row["rnd_mean"] = float(rnd_mean) if rnd_mean is not None else None
    new_row["all_pass"] = float(all_pass) if all_pass is not None else None
    new_row["avg_rounds"] = (
        sum(n_rounds_per_prompt) / max(len(n_rounds_per_prompt), 1)
    )
    # avg_commit_per_round only defined where the lane has it (skip for strict).
    if r_existing.get("avg_commit_per_round") is not None:
        new_row["avg_commit_per_round"] = (
            sum(n_committed_per_round) / max(len(n_committed_per_round), 1)
        )
    return new_row


def _render_md(new_rows: List[Dict], existing: Dict, prompt_indices: List[int],
               decode_s: float, offline_s: float) -> str:
    def _fmt_opt(v, w=10, p=4):
        if v is None:
            return f"{'—':>{w}}"
        return f"{float(v):>{w}.{p}f}"

    def _fmt_signed(v, w=10, p=4):
        if v is None:
            return f"{'—':>{w}}"
        return f"{float(v):>+{w}.{p}f}"

    md: List[str] = []
    md.append(
        f"# Phase 26 — lossy SD baseline + confidence commit rule "
        f"({existing['n_prompts']} prompts, "
        f"max_new_tokens={existing['max_new_tokens']}, "
        f"γ={existing['gamma']}, T={existing['T']}, temperature=0)\n\n"
    )
    md.append(
        "tok/s measured inside the decode loop. NLL, ΔNLL, tok_succ, "
        "rnd_mean, all_pass are computed on the stored decode JSONs — "
        "tok_succ/rnd_mean/all_pass inline during decode (temp=0: "
        "argmax(p)==draft), NLL in an offline pass. Lossy-l commits "
        "L_lossy + 1 per round (bonus/fallback included, matches the "
        "original SD paper). Predictor threshold and confidence lanes "
        "commit max(1, L̂) per round (no bonus). rnd_mean and all_pass "
        "use an ALL-ROUNDS denominator for every lane (including lossy "
        "rounds where L_lossy=0).\n\n"
    )
    md.append("```\n")
    header = (
        f"  {'family':<14}{'rule':<12}{'thresh':>10}"
        f"{'tok/s':>8}{'NLL':>9}{'ΔNLL':>10}"
        f"{'tok_succ':>10}{'rnd_mean':>10}{'all_pass':>10}\n"
    )
    md.append(header)
    for r in new_rows:
        md.append(
            f"  {r['family']:<14}{r['rule']:<12}"
            f"{r['threshold_label']:>10}"
            f"{float(r['tok_s_mean']):>8.2f}"
            f"{_fmt_opt(r.get('NLL'), 9, 4)}"
            f"{_fmt_signed(r.get('delta_NLL'), 10, 4)}"
            f"{_fmt_opt(r.get('tok_succ'), 10, 4)}"
            f"{_fmt_opt(r.get('rnd_mean'), 10, 4)}"
            f"{_fmt_opt(r.get('all_pass'), 10, 4)}\n"
        )
    md.append("```\n\n")
    if prompt_indices:
        md.append(
            f"Prompt indices: {prompt_indices[0]}..{prompt_indices[-1]} "
            f"({len(prompt_indices)} prompts)\n"
        )
    md.append(f"Decode wallclock (sum): {decode_s:.1f}s\n")
    md.append(f"Offline NLL+success wallclock (sum): {offline_s:.1f}s\n")
    return "".join(md)


def _rebuild_unified_table(out_dir: Path) -> Tuple[List[Dict], List[Dict]]:
    """Re-aggregate per-prompt → rows → unified_table.{md,json}.

    Returns (old_rows, new_rows) for before/after reporting.
    """
    ut_path = out_dir / "unified_table.json"
    with open(ut_path) as f:
        existing = json.load(f)
    old_rows = list(existing["rows"])

    new_rows: List[Dict] = []
    for r in existing["rows"]:
        fp = out_dir / _filename_for_row(r)
        with open(fp) as f:
            lane = json.load(f)
        if r["family"] == "strict":
            # Strict rows carry no tok_succ/rnd_mean/all_pass by design;
            # keep the existing row, but refresh avg_rounds from disk.
            new_r = dict(r)
            pp = lane["per_prompt"]
            n_rounds_per_prompt = [len(p.get("rounds", [])) for p in pp]
            new_r["avg_rounds"] = (
                sum(n_rounds_per_prompt) / max(len(n_rounds_per_prompt), 1)
            )
            new_r["tok_s_mean"] = float(lane.get("tok_s_mean", r["tok_s_mean"]))
            new_rows.append(new_r)
        else:
            new_rows.append(_aggregate_row(lane, r))

    timings = existing.get("timings_s") or {}
    decode_s = sum((timings.get("decode") or {}).values()) if timings else 0.0
    offline_s = sum((timings.get("offline") or {}).values()) if timings else 0.0

    md_text = _render_md(
        new_rows, existing, existing.get("prompt_indices", []),
        decode_s, offline_s,
    )

    # Write outputs.
    with open(out_dir / "unified_table.md", "w") as f:
        f.write(md_text)
    new_existing = dict(existing)
    new_existing["rows"] = new_rows
    new_existing["correction_note"] = (
        "lossy rnd_mean/all_pass corrected to all-rounds denominator "
        "(tok_succ unchanged by construction)"
    )
    with open(out_dir / "unified_table.json", "w") as f:
        json.dump(new_existing, f, indent=2, default=str)
    return old_rows, new_rows


def _print_diff(old_rows: List[Dict], new_rows: List[Dict]) -> None:
    """Print lossy before/after rows side by side."""
    old_by_key = {r["lane_key"]: r for r in old_rows}
    lossy = [r for r in new_rows if r["family"] == "lossy"]

    def _v(x, fmt="7.4f"):
        return "—" if x is None else format(float(x), fmt)

    print("\n--- LOSSY ROWS BEFORE vs AFTER ---")
    print(f"{'label':<10} {'metric':<10} {'before':>10} {'after':>10}")
    for r in lossy:
        old = old_by_key[r["lane_key"]]
        lab = r["threshold_label"]
        for m in ("tok_succ", "rnd_mean", "all_pass"):
            print(f"{lab:<10} {m:<10} "
                  f"{_v(old.get(m)):>10} {_v(r.get(m)):>10}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out_dir", type=str,
        default="outputs/phase26_sw_12140107",
        help="Directory holding online_*.json and unified_table.json.",
    )
    ap.add_argument(
        "--skip_plots", action="store_true",
        help="Skip invoking phase26_plots.py.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (_REPO_ROOT / out_dir).resolve()
    else:
        out_dir = out_dir.resolve()

    print(f"[recompute] out_dir = {out_dir}")

    lossy_jsons = sorted(out_dir.glob("online_lossy_l_*.json"))
    if not lossy_jsons:
        print("[recompute] no online_lossy_l_*.json found — nothing to do")
        return 1
    print(f"[recompute] correcting {len(lossy_jsons)} lossy JSONs")
    for fp in lossy_jsons:
        summary = _correct_lossy_json(fp)
        print(
            f"[recompute]   {fp.name}: "
            f"rnd_mean {summary['rnd_mean_before']:.4f} → "
            f"{summary['rnd_mean_after']:.4f}  "
            f"all_pass {summary['all_pass_before']:.4f} → "
            f"{summary['all_pass_after']:.4f}  "
            f"tok_succ {summary['tok_succ_before']:.4f} → "
            f"{summary['tok_succ_after']:.4f}"
        )

    print("[recompute] rebuilding unified_table.{md,json}")
    old_rows, new_rows = _rebuild_unified_table(out_dir)
    _print_diff(old_rows, new_rows)

    if not args.skip_plots:
        print("[recompute] invoking phase26_plots.py for plots + condensed table")
        subprocess.run(
            [
                sys.executable,
                str(_REPO_ROOT / "scripts/phase26_plots.py"),
                "--table_json", str(out_dir / "unified_table.json"),
                "--out_dir", str(out_dir),
            ],
            check=True,
        )

    print("[recompute] DONE.")
    print(f"[recompute] wrote {out_dir}/unified_table.md")
    print(f"[recompute] wrote {out_dir}/unified_table.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

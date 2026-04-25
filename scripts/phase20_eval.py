"""Phase 20 — pure evaluation pass.

No training. For each method, produce:
  Table A (predictor-level, per-position-aware MAE)
    - Overall MAE (survived-masked)
    - Per-position MAE_j for j=0..γ−1
    - MAE_accepted      : masked to j < record.L  (positions strict ACCEPTED)
    - MAE_through_rej   : masked to j <= record.L (== survived mask; sanity)
    - MAE_prefix_weight : survived-masked, w_j = 1 / (1 + j), sum-normalised
  Table B (system-level, per method × τ)
    - offline_cf_at_1 (from preds_test.pt + stored record.L)
    - online_cf_at_1 (from online_acceptance_tau_*.json)
    - tok_s (from online JSON)
    - verifier NLL on the method's ONLINE-generated new tokens
    - ΔNLL = NLL_method - NLL_strict_on_original_drafter_ref
  Figure 1 (pareto): x = ΔNLL, y = tok/s, markers per (method, τ)

Strict reference: strict SpecDiff under the ORIGINAL pretrained drafter
(stage1_pp.pt records), prompts 60..69, first `max_new_tokens` new tokens
reconstructed by walking stored records. This is the fixed baseline that
all methods are compared against — so "ΔNLL" measures total degradation
relative to the canonical strict run, regardless of whether a method
fine-tuned its drafter.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from accpre.core.commit import commit_threshold
from accpre.core.protocol import ProtocolConfig
from accpre.core.schema import load_records
from accpre.data.splits import POOL_SIZE, TRAIN_N, VAL_N


# Method registry — (name, preds_test_pt, online_json_dir). Preds paths
# are relative to repo root; online JSONs are the phase-specific outputs
# we already have on disk.
METHODS = [
    {
        "name": "frozen_1A",
        "preds": "checkpoints/acc_1a/preds_test.pt",
        "online_dir": "outputs/phase12_1a1b_11830736/acc_1a",
    },
    {
        "name": "lazy_jnt",
        "preds": "checkpoints/acc_jnt_1a/preds_test.pt",
        "online_dir": "outputs/phase15_jnt1a_11859744/online",
    },
    {
        "name": "live_jnt_5ep",
        "preds": "checkpoints/acc_jnt_1a_liveq2/preds_test.pt",
        "online_dir": "outputs/phase16_liveq2_11860043/online",
    },
    {
        "name": "live_jnt_10ep",
        "preds": "checkpoints/acc_jnt_1a_liveq2_long/preds_test.pt",
        "online_dir": "outputs/phase17_liveq2_long_11860808/online",
    },
]
TAUS = (0.5, 0.7, 0.9)


def _load_protocol(path: Path) -> ProtocolConfig:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    for k in ("draft_salt", "accept_salt", "fallback_salt",
              "schema_version", "max_verifier_ctx"):
        if k in d and isinstance(d[k], str):
            d[k] = int(d[k], 0)
    return ProtocolConfig(**d)


# ----------------------------------------------------------------------
# Part A — predictor-level metrics
# ----------------------------------------------------------------------


def _load_test_records(root: Path, protocol: ProtocolConfig) -> List:
    all_recs = load_records(
        str(root / "data_collected/stage1_pp.pt"),
        expected_protocol=protocol,
    )
    test_p = set(range(TRAIN_N + VAL_N, POOL_SIZE))
    return [r for r in all_recs if r.prompt_idx in test_p and r.gamma == 8]


def predictor_table(
    root: Path, test_records: List, gamma: int = 8,
) -> List[Dict]:
    """Compute per-method predictor-quality metrics on the stored
    test-split records. All methods share the same (q2_target, survived,
    record.L) references — only the model's q2_hat differs.
    """
    rows: List[Dict] = []
    for m in METHODS:
        preds_path = root / m["preds"]
        if not preds_path.is_file():
            print(f"[phase20] skip {m['name']}: {preds_path} missing")
            continue
        preds = torch.load(str(preds_path), weights_only=False)
        # Align by record_idx.
        preds = sorted(preds, key=lambda r: int(r["record_idx"]))
        assert len(preds) == len(test_records), (
            f"{m['name']}: preds={len(preds)} vs records={len(test_records)}"
        )

        # Per-record arrays aligned by record_idx.
        row: Dict = {"method": m["name"]}

        # Flatten (n_records, γ) matrices.
        q_hat_mat: List[List[float]] = []
        q2_mat:    List[List[float]] = []
        surv_mat:  List[List[int]]   = []
        L_vec:     List[int]         = []
        for p, r in zip(preds, test_records):
            q_hat_mat.append([float(x) for x in p["q2_hat"]])
            q2_mat.append([float(x) for x in p["q2_target"]])
            surv_mat.append([int(x) for x in p["survived"]])
            L_vec.append(int(r.L))
        q_hat = np.asarray(q_hat_mat, dtype=np.float64)   # (N, γ)
        q2    = np.asarray(q2_mat,    dtype=np.float64)   # (N, γ)
        surv  = np.asarray(surv_mat,  dtype=np.int64)     # (N, γ)
        L_arr = np.asarray(L_vec,     dtype=np.int64)     # (N,)
        N, G = q_hat.shape

        abs_err = np.abs(q_hat - q2)                     # (N, γ)
        survived_mask = (surv == 1).astype(np.float64)

        # Overall MAE (survived-masked — matches Phase 12 reporting).
        denom = survived_mask.sum()
        row["MAE_overall"] = float(
            (abs_err * survived_mask).sum() / max(denom, 1e-9)
        )

        # Per-j MAE.
        per_j_mae: List[float] = []
        per_j_n: List[int] = []
        for j in range(G):
            m_j = survived_mask[:, j]
            n_j = int(m_j.sum())
            if n_j == 0:
                per_j_mae.append(float("nan"))
                per_j_n.append(0)
                continue
            per_j_mae.append(float((abs_err[:, j] * m_j).sum() / n_j))
            per_j_n.append(n_j)
        row["MAE_per_j"] = per_j_mae
        row["n_per_j"]   = per_j_n

        # MAE_accepted: j < record.L.
        # survived is 1 for j <= L_strict; "accepted" is the strict prefix
        # j < L_strict (exclusive of the rejection point). For L=0 records
        # the set is empty and they contribute nothing.
        j_idx = np.arange(G)[None, :]                     # (1, γ)
        L_bc  = L_arr[:, None]                            # (N, 1)
        accepted_mask = (j_idx < L_bc).astype(np.float64)
        denom_a = accepted_mask.sum()
        row["MAE_accepted"] = (
            float((abs_err * accepted_mask).sum() / max(denom_a, 1e-9))
            if denom_a > 0 else float("nan")
        )
        row["n_accepted"] = int(denom_a)

        # MAE_through_rej: j <= record.L. Trivially equal to survived-MAE;
        # reported for sanity (should match MAE_overall).
        through_mask = (j_idx <= L_bc).astype(np.float64) * survived_mask
        denom_t = through_mask.sum()
        row["MAE_through_rej"] = (
            float((abs_err * through_mask).sum() / max(denom_t, 1e-9))
        )

        # MAE_prefix_weighted: w_j = 1/(1+j) on survived positions.
        w = 1.0 / (1.0 + j_idx.astype(np.float64))        # (1, γ)
        w_mask = survived_mask * w
        denom_w = w_mask.sum()
        row["MAE_prefix_weighted"] = (
            float((abs_err * w_mask).sum() / max(denom_w, 1e-9))
        )

        # Distribution stats on the flattened survived q_hat (handy for
        # cross-referencing with earlier reports).
        qh_flat = q_hat[surv == 1]
        q2_flat = q2[surv == 1]
        row["Qhat_mean"] = float(qh_flat.mean())
        row["Qhat_std"]  = float(qh_flat.std(ddof=0))
        row["Qhat_max"]  = float(qh_flat.max())
        row["Q2_std"]    = float(q2_flat.std(ddof=0))

        # Offline CF@1 via commit_threshold(Q̂, τ) on same stored records.
        cfs: Dict[str, Dict] = {}
        for tau in TAUS:
            exact = under = over = 0
            lhat0 = 0
            for p, r in zip(preds, test_records):
                q_list = [float(x) for x in p["q2_hat"]]
                L_hat = commit_threshold(q_list, float(tau))
                L_hat = max(0, min(L_hat, r.gamma))
                if L_hat == 0:
                    lhat0 += 1
                if L_hat == int(r.L):
                    exact += 1
                elif L_hat < int(r.L):
                    under += 1
                else:
                    over += 1
            cfs[str(float(tau))] = {
                "offline_cf": exact / N,
                "offline_under": under / N,
                "offline_over":  over / N,
                "offline_Lhat_eq_0_frac": lhat0 / N,
            }
        row["offline_per_tau"] = cfs
        rows.append(row)
    return rows


# ----------------------------------------------------------------------
# Part B — verifier NLL on generated text
# ----------------------------------------------------------------------


def _reconstruct_strict_new_tokens(
    root: Path, protocol: ProtocolConfig, prompt_indices: List[int],
    max_new_tokens: int, gamma: int = 8,
) -> Dict[int, Tuple[List[int], List[int]]]:
    """For each prompt idx, return (prefix_ids_list, strict_new_token_list)
    under the ORIGINAL drafter, truncated to `max_new_tokens` new tokens.

    The trajectory is reconstructed from stage1_pp.pt by walking records
    in round order and appending `draft_tokens[:L] + [bonus]` per round,
    stopping after `max_new_tokens` new tokens.
    """
    from accpre.data.prompts import load_owt_prompts
    from accpre.data.splits import POOL_SIZE, PREFIX_LEN, PROMPT_SEED

    pool = load_owt_prompts(
        n_prompts=POOL_SIZE, prefix_len=PREFIX_LEN, seed=PROMPT_SEED,
    )
    all_recs = load_records(
        str(root / "data_collected/stage1_pp.pt"),
        expected_protocol=protocol,
    )
    # Build prompt → sorted records.
    from collections import defaultdict
    by_prompt = defaultdict(list)
    for r in all_recs:
        if r.gamma == gamma:
            by_prompt[int(r.prompt_idx)].append(r)
    for p in by_prompt:
        by_prompt[p].sort(key=lambda rr: int(rr.round_idx))

    out: Dict[int, Tuple[List[int], List[int]]] = {}
    for p_idx in prompt_indices:
        prefix_ids = [int(x) for x in pool[p_idx][0].tolist()]
        new_tokens: List[int] = []
        for rec in by_prompt[p_idx]:
            take = rec.draft_tokens[: int(rec.L)]
            new_tokens.extend(int(t) for t in take)
            new_tokens.append(int(rec.bonus_or_fallback_token))
            if len(new_tokens) >= max_new_tokens:
                break
        new_tokens = new_tokens[:max_new_tokens]
        out[p_idx] = (prefix_ids, new_tokens)
    return out


def score_nll_on_new_tokens(
    verifier, token_seq: List[int], n_prefix: int, device,
) -> Tuple[float, int]:
    """Run verifier on the full seq, return (sum_nll_on_new_tokens, n_new).

    For position i in [n_prefix, len(seq)):
        log p(x_i | x_<i) = verifier_log_probs[i-1, x_i]
    NLL = -sum log p. We return the SUM (not mean) so the caller can
    aggregate across prompts with proper weighting.
    """
    if len(token_seq) <= n_prefix:
        return 0.0, 0
    x = torch.tensor(token_seq, dtype=torch.long, device=device)
    with torch.no_grad():
        log_probs = verifier.score(x).to(torch.float32)     # (L, V)
    n_new = len(token_seq) - n_prefix
    idx_logits = torch.arange(n_prefix - 1, n_prefix + n_new - 1, device=device)
    idx_tokens = torch.tensor(
        token_seq[n_prefix: n_prefix + n_new], dtype=torch.long, device=device,
    )
    selected = log_probs[idx_logits, idx_tokens]             # (n_new,)
    nll_sum = float(-selected.sum().item())
    return nll_sum, int(n_new)


def online_and_nll_table(
    root: Path, protocol: ProtocolConfig, max_new_tokens: int,
) -> Dict:
    """Compute per (method, τ):
      - online_cf_at_1, tok_s (from JSON)
      - verifier NLL on the method's online-generated new tokens
      - ΔNLL vs the fixed strict reference
    """
    # Build strict ref once (all 10 online prompts: 60..69).
    from accpre.models.verifier_gpt2 import GPT2Verifier
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {
        "float32": torch.float32, "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[protocol.dtype]
    verifier = GPT2Verifier(
        model_name=protocol.verifier_model, device=device, dtype=dtype,
    )

    prompt_indices = list(range(TRAIN_N + VAL_N, TRAIN_N + VAL_N + 10))
    print(f"[phase20] building strict reference for prompts {prompt_indices}")
    strict_traj = _reconstruct_strict_new_tokens(
        root, protocol, prompt_indices, max_new_tokens,
    )

    # Score strict ref.
    strict_nll_sum = 0.0
    strict_n = 0
    strict_per_prompt: Dict[int, Dict] = {}
    for p_idx, (prefix_ids, new_tokens) in strict_traj.items():
        seq = prefix_ids + new_tokens
        nll_s, n_s = score_nll_on_new_tokens(
            verifier, seq, n_prefix=len(prefix_ids), device=device,
        )
        strict_nll_sum += nll_s
        strict_n += n_s
        strict_per_prompt[p_idx] = {
            "n_new": n_s,
            "nll_mean": (nll_s / n_s) if n_s > 0 else float("nan"),
        }
    strict_nll_mean = strict_nll_sum / max(strict_n, 1)
    print(
        f"[phase20] strict ref: total new tokens={strict_n} "
        f"mean NLL/tok={strict_nll_mean:.4f}"
    )

    # For each method × τ, load online JSON, score, compute NLL + ΔNLL.
    results = {"strict_ref": {
        "nll_mean": strict_nll_mean,
        "n_new_total": strict_n,
        "per_prompt": strict_per_prompt,
    }, "methods": {}}

    for m in METHODS:
        online_dir = root / m["online_dir"]
        if not online_dir.is_dir():
            print(f"[phase20] skip {m['name']}: {online_dir} missing")
            continue
        results["methods"][m["name"]] = {}
        for tau in TAUS:
            tau_tag = f"0p{int(tau * 10):d}"
            f = online_dir / f"online_acceptance_tau_{tau_tag}.json"
            if not f.is_file():
                print(f"[phase20] skip {m['name']} tau={tau}: missing {f}")
                continue
            with open(f) as fh:
                d = json.load(fh)
            l3 = d["l3"]
            online_cf = float(l3["cf_at_1_aggregate"])
            tok_s_mean = float(l3["tok_s_mean"])
            # Score each per-prompt generated sequence.
            method_nll_sum = 0.0
            method_n = 0
            per_prompt: List[Dict] = []
            for p in d["online_lane"]["per_prompt"]:
                gen = [int(x) for x in p["generated_ids"]]
                n_new = int(p["n_new_tokens"])
                n_prefix = len(gen) - n_new
                nll_s, n_s = score_nll_on_new_tokens(
                    verifier, gen, n_prefix=n_prefix, device=device,
                )
                method_nll_sum += nll_s
                method_n += n_s
                per_prompt.append({
                    "prompt_idx": int(p["prompt_idx"]),
                    "n_new": n_s,
                    "nll_mean": (nll_s / n_s) if n_s > 0 else float("nan"),
                    "tok_s": float(p["tok_s"]),
                })
            method_nll_mean = method_nll_sum / max(method_n, 1)
            results["methods"][m["name"]][str(tau)] = {
                "tau": float(tau),
                "online_cf_at_1": online_cf,
                "tok_s_mean": tok_s_mean,
                "n_new_total": method_n,
                "nll_mean": method_nll_mean,
                "delta_nll_vs_strict": method_nll_mean - strict_nll_mean,
                "per_prompt": per_prompt,
            }
            print(
                f"[phase20] {m['name']:<18} τ={tau}  "
                f"CF@1={online_cf:.3f}  tok/s={tok_s_mean:.2f}  "
                f"NLL={method_nll_mean:.4f}  ΔNLL={method_nll_mean - strict_nll_mean:+.4f}"
            )
    return results


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------


def _fmt_table_a(rows: List[Dict]) -> str:
    lines = ["# Table A — predictor-quality\n\n"]
    lines.append("```\n")
    hdr = (
        f"  {'method':<16}{'MAE_over':>10}{'MAE_acc':>10}{'MAE_wgt':>10}"
    )
    for j in range(8):
        hdr += f"{'j=' + str(j):>8}"
    hdr += "\n"
    lines.append(hdr)
    for row in rows:
        line = (
            f"  {row['method']:<16}"
            f"{row['MAE_overall']:>10.4f}"
            f"{row['MAE_accepted']:>10.4f}"
            f"{row['MAE_prefix_weighted']:>10.4f}"
        )
        for j, v in enumerate(row["MAE_per_j"]):
            if not np.isfinite(v):
                line += f"{'  -':>8}"
            else:
                line += f"{v:>8.4f}"
        lines.append(line + "\n")
    lines.append("\n  n_per_j (survived count at each position):\n")
    lines.append(f"  {'':16}{'':10}{'':10}{'':10}")
    # Assume all rows share the same n_per_j (they should, same stored records).
    if rows:
        for n in rows[0]["n_per_j"]:
            lines[-1] += f"{n:>8d}"
        lines[-1] += "\n"
    lines.append(f"  n_accepted (shared): {rows[0]['n_accepted'] if rows else 0}\n")
    lines.append("```\n\n")
    return "".join(lines)


def _fmt_table_b(rows: List[Dict], nll_results: Dict) -> str:
    lines = ["# Table B — system-quality (per method × τ)\n\n"]
    lines.append(f"strict ref NLL = {nll_results['strict_ref']['nll_mean']:.4f} "
                 f"on {nll_results['strict_ref']['n_new_total']} new tokens "
                 f"(prompts 60–69, first ≤64 each).\n\n")
    lines.append("```\n")
    lines.append(
        f"  {'method':<16}{'τ':>6}{'offCF1':>8}{'onCF1':>8}"
        f"{'tok/s':>8}{'NLL':>10}{'ΔNLL':>10}{'offU':>7}{'offO':>7}\n"
    )
    for row in rows:
        name = row["method"]
        for tau in TAUS:
            tau_key = str(float(tau))
            off = row.get("offline_per_tau", {}).get(tau_key, {})
            m = nll_results.get("methods", {}).get(name, {}).get(str(tau))
            if m is None:
                on_cf = tok = nll = dnll = float("nan")
            else:
                on_cf = m["online_cf_at_1"]
                tok = m["tok_s_mean"]
                nll = m["nll_mean"]
                dnll = m["delta_nll_vs_strict"]
            line = (
                f"  {name:<16}{float(tau):>6.1f}"
                f"{off.get('offline_cf', float('nan')):>8.4f}"
                f"{on_cf:>8.4f}"
                f"{tok:>8.2f}"
                f"{nll:>10.4f}"
                f"{dnll:>10.4f}"
                f"{off.get('offline_under', float('nan')):>7.3f}"
                f"{off.get('offline_over', float('nan')):>7.3f}\n"
            )
            lines.append(line)
    lines.append("```\n")
    return "".join(lines)


def _maybe_pareto_figure(rows: List[Dict], nll_results: Dict, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[phase20] matplotlib unavailable ({e}); skipping figure.")
        return
    colors = {
        "frozen_1A":       "#4c78a8",
        "lazy_jnt":        "#a0a0a0",
        "live_jnt_5ep":    "#54a24b",
        "live_jnt_10ep":   "#e45756",
    }
    markers_by_tau = {0.5: "o", 0.7: "s", 0.9: "^"}
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for row in rows:
        name = row["method"]
        color = colors.get(name, "black")
        for tau in TAUS:
            m = nll_results.get("methods", {}).get(name, {}).get(str(tau))
            if m is None:
                continue
            x = m["delta_nll_vs_strict"]
            y = m["tok_s_mean"]
            ax.scatter(x, y, s=90, color=color, marker=markers_by_tau[tau],
                       label=f"{name} τ={tau}")
            ax.annotate(f"{name}\nτ={tau}", (x, y), textcoords="offset points",
                        xytext=(6, 6), fontsize=7, color=color, alpha=0.8)
    ax.axvline(0.0, linestyle="--", color="#cccccc", linewidth=1, label="strict ref")
    ax.set_xlabel("Δ NLL vs strict reference  (↓ better)")
    ax.set_ylabel("tok/s  (↑ better)")
    ax.set_title("Phase 20 Pareto: throughput vs quality gap to strict")
    ax.grid(True, alpha=0.3)
    # Legend is too crowded with per-point labels; show only markers-by-τ.
    from matplotlib.lines import Line2D
    legend_h = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
               label="τ = 0.5", markersize=8),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="black",
               label="τ = 0.7", markersize=8),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="black",
               label="τ = 0.9", markersize=8),
    ]
    for name, color in colors.items():
        legend_h.append(Line2D([0], [0], marker="o", color="w",
                               markerfacecolor=color, label=name,
                               markersize=8))
    ax.legend(handles=legend_h, loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=130)
    plt.close(fig)
    print(f"[phase20] wrote {out_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=str, default=str(_REPO_ROOT))
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--max_new_tokens", type=int, default=64,
                    help="Match the online-decode budget.")
    ap.add_argument("--skip_nll", action="store_true",
                    help="Only compute Table A (no GPU needed).")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    protocol = _load_protocol(root / "configs/protocol.yaml")
    test_records = _load_test_records(root, protocol)
    print(
        f"[phase20] n_test_records={len(test_records)}  protocol="
        f"{protocol.fingerprint()[:60]}..."
    )

    # Part A.
    print("\n=== Part A: predictor-level metrics ===")
    rows_a = predictor_table(root, test_records)
    with open(out_dir / "predictor_table.json", "w") as f:
        json.dump(rows_a, f, indent=2, default=str)
    rep_a = _fmt_table_a(rows_a)
    with open(out_dir / "predictor_table.md", "w") as f:
        f.write(rep_a)
    print(rep_a)

    if args.skip_nll:
        print("[phase20] --skip_nll set; Table A only. Done.")
        return 0

    # Part B.
    print("\n=== Part B: verifier NLL / ΔNLL ===")
    nll_results = online_and_nll_table(root, protocol, args.max_new_tokens)
    with open(out_dir / "system_table.json", "w") as f:
        json.dump(nll_results, f, indent=2, default=str)
    rep_b = _fmt_table_b(rows_a, nll_results)
    with open(out_dir / "system_table.md", "w") as f:
        f.write(rep_b)
    print(rep_b)

    # Part C — Pareto figure.
    print("\n=== Part C: Pareto figure ===")
    _maybe_pareto_figure(rows_a, nll_results, out_dir / "pareto.png")

    # Combined report.
    combined = (
        "# Phase 20 — new evaluation metrics\n\n"
        + rep_a + "\n" + rep_b
        + "\n## Figure\n\n![pareto](pareto.png)\n"
    )
    with open(out_dir / "report.md", "w") as f:
        f.write(combined)
    print(f"[phase20] wrote {out_dir}/report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())

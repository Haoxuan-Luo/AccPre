"""Executable oracle-Q2 baseline.

Per round:
  1. drafter.draft(running, γ=8, T=2) → draft_tokens, draft_log_probs
  2. verifier.score(concat(running, draft_tokens)) → target_log_probs
     (same verifier cost as strict — full prefix+draft forward per round)
  3. For each j: q = exp(draft_log_probs[j, tok_j]);
                p = exp(target_log_probs[prefix_len-1+j, tok_j]);
                Q2_j = min(1, p / q).
  4. L_hat = commit_threshold(Q2, τ).
  5. Commit max(1, L_hat) draft tokens. NO strict fallback/bonus.
  6. running := running ++ committed.

NLL is accumulated on the committed tokens using the same
`target_log_probs` computed for the Q2 decision in step 2 — the verifier
conditional is correct because draft_tokens were concatenated to running
before scoring, so `target_log_probs[prefix_len-1+j]` is the verifier's
conditional on (running, draft_tokens[:j]) by teacher-forcing.

Implementation note: this uses the ORIGINAL pretrained MDLM drafter
(same state as strict). Trajectories diverge from strict after round 0
because the commit rule differs, so "online CF@1 vs stored record.L"
is only meaningful on round 0 — for the full trajectory the right
offline CF@1 reading is the stored-record version from the Phase 14
oracle_q2_ceiling table, which is reported in the final summary as the
same-stored-data ceiling. The NEW numbers here are tok/s, NLL, ΔNLL,
all measured on the actual rollout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from accpre.core.commit import commit_threshold
from accpre.core.draft_verify import _make_generator
from accpre.core.protocol import ProtocolConfig, derive_seed
from accpre.data.splits import POOL_SIZE, TRAIN_N, VAL_N, test_split


EPS = 1e-10
STRICT_REF_NLL = 2.8932      # Phase 20 main output on same prompts/budget.


def _load_protocol(path: Path) -> ProtocolConfig:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    for k in ("draft_salt", "accept_salt", "fallback_salt",
              "schema_version", "max_verifier_ctx"):
        if k in d and isinstance(d[k], str):
            d[k] = int(d[k], 0)
    return ProtocolConfig(**d)


@torch.no_grad()
def run_oracle_lane(
    drafter, verifier, prompts, prompt_indices,
    protocol: ProtocolConfig,
    gamma: int, T: int, max_new_tokens: int, tau: float,
) -> Dict:
    """One oracle-Q2 rollout lane at a fixed τ.

    Returns a dict with per-prompt stats + aggregates.
    """
    device = drafter.device
    MAX_CTX = protocol.max_verifier_ctx
    per_prompt: List[Dict] = []
    agg_sum_log_p = 0.0       # sum of verifier log p on committed tokens
    agg_n_scored = 0
    agg_total_elapsed = 0.0
    agg_total_new = 0

    for i, (prefix_ids, _text) in enumerate(prompts):
        p_idx = int(prompt_indices[i])
        running = prefix_ids.to(device).clone()
        prefix_start_len = int(running.shape[0])
        round_idx = 0
        total_log_p = 0.0
        total_new = 0
        Lhat_counts: Dict[int, int] = {}
        t0 = time.time()

        while (running.shape[0] - prefix_start_len) < max_new_tokens:
            if running.shape[0] >= MAX_CTX:
                break
            remaining = max_new_tokens - (running.shape[0] - prefix_start_len)
            ctx_room = MAX_CTX - int(running.shape[0])
            cur_gamma = min(int(gamma), int(remaining), int(ctx_room))
            if cur_gamma <= 0:
                break

            seed = derive_seed(protocol, p_idx, round_idx)
            draft_rng = _make_generator(device, seed ^ protocol.draft_salt)

            draft_tokens, draft_log_probs = drafter.draft(
                prefix_ids=running, gamma=cur_gamma, T=T,
                temperature=protocol.temperature, q_mode=protocol.q_mode,
                generator=draft_rng,
            )

            # Verifier forward on (prefix + draft) — same cost as strict.
            candidate = torch.cat([running, draft_tokens])
            target_log_probs = verifier.score(candidate).to(torch.float32)

            prefix_len = int(running.shape[0])
            idx = torch.arange(cur_gamma, device=device)
            q_j = draft_log_probs[idx, draft_tokens.long()].exp().clamp(min=EPS)
            p_j = target_log_probs[
                prefix_len - 1 + idx, draft_tokens.long()
            ].exp().clamp(min=EPS)
            q2 = torch.minimum(torch.ones_like(p_j), p_j / q_j)
            q2_list = [float(x) for x in q2.cpu().tolist()]

            L_hat = commit_threshold(q2_list, float(tau))
            L_hat = max(0, min(L_hat, cur_gamma))
            Lhat_counts[int(L_hat)] = Lhat_counts.get(int(L_hat), 0) + 1
            n_commit = max(1, L_hat)
            committed = draft_tokens[:n_commit]

            # NLL contribution from this round — teacher-forcing log-probs
            # for the committed tokens, under the verifier. The conditional
            # context is correct because draft_tokens were concatenated to
            # running before scoring.
            for j in range(n_commit):
                lp = float(
                    target_log_probs[prefix_len - 1 + j,
                                     draft_tokens[j].long()].item()
                )
                total_log_p += lp
                total_new += 1

            running = torch.cat([running, committed])
            round_idx += 1

        elapsed = time.time() - t0
        n_new = int(running.shape[0]) - prefix_start_len
        tok_s = n_new / max(elapsed, 1e-9)
        nll_mean = -total_log_p / max(total_new, 1)
        per_prompt.append({
            "prompt_idx": p_idx,
            "n_new_tokens": int(n_new),
            "elapsed_s": float(elapsed),
            "tok_s": float(tok_s),
            "generated_ids": [int(x) for x in running.tolist()],
            "nll_mean": float(nll_mean),
            "n_scored": int(total_new),
            "Lhat_dist": {str(k): int(v) for k, v in Lhat_counts.items()},
        })
        agg_sum_log_p += total_log_p
        agg_n_scored += total_new
        agg_total_elapsed += elapsed
        agg_total_new += n_new

    agg_tok_s = sum(p["tok_s"] for p in per_prompt) / max(len(per_prompt), 1)
    agg_nll_mean = -agg_sum_log_p / max(agg_n_scored, 1)
    return {
        "tau": float(tau),
        "tok_s_mean": float(agg_tok_s),
        "nll_mean": float(agg_nll_mean),
        "n_scored_total": int(agg_n_scored),
        "n_new_total": int(agg_total_new),
        "elapsed_total_s": float(agg_total_elapsed),
        "per_prompt": per_prompt,
    }


def _pareto_figure(
    points: List[Dict], out_path: Path,
    strict_point: Tuple[float, float],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[oracle_exec] matplotlib unavailable ({e})")
        return
    colors = {
        "frozen_1A":    "#4c78a8",
        "live_jnt_5ep": "#54a24b",
        "strict":       "#000000",
        "oracle_exec":  "#f58518",
    }
    markers_by_tau = {0.5: "o", 0.7: "s", 0.9: "^", "—": "D"}
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for pt in points:
        name = pt["method"]
        tau = pt["tau"]
        tau_key = "—" if isinstance(tau, str) else float(tau)
        marker = markers_by_tau.get(tau_key, "x")
        color = colors.get(name, "#888")
        x = pt["delta_NLL"]
        y = pt["tok_s"]
        ax.scatter(
            x, y,
            s=130 if name == "strict" else (120 if name == "oracle_exec" else 90),
            color=color, marker=marker,
            edgecolor="black" if name in ("strict", "oracle_exec") else None,
            linewidth=1.2 if name in ("strict", "oracle_exec") else 0.0,
        )
        label = "strict" if name == "strict" else f"{name}\nτ={tau}"
        ax.annotate(label, (x, y),
                    textcoords="offset points", xytext=(7, 7),
                    fontsize=7.5, color=color, alpha=0.9)
    ax.axvline(0.0, linestyle="--", color="#bbbbbb", linewidth=1)
    ax.axhline(strict_point[1], linestyle=":", color="#bbbbbb", linewidth=1)
    ax.set_xlabel("Δ NLL vs strict reference  (↓ better, strict at 0)")
    ax.set_ylabel("tok/s  (↑ better)")
    ax.set_title(
        "Phase 20 Pareto + executable oracle-Q2\n"
        "(strict diamond; oracle-exec squares with black edge)"
    )
    ax.grid(True, alpha=0.3)
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
        legend_h.append(Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=color, label=name, markersize=8,
            markeredgecolor=("black" if name in ("strict", "oracle_exec") else color),
        ))
    ax.legend(handles=legend_h, loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=130)
    plt.close(fig)
    print(f"[oracle_exec] wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--n_prompts", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--taus", type=str, default="0.5,0.7,0.9")
    ap.add_argument("--phase20_dir", type=str,
                    default="outputs/phase20_eval_11869681")
    ap.add_argument("--oracle_json", type=str,
                    default="outputs/oracle_q2_ceiling/summary.json")
    ap.add_argument("--strict_json", type=str,
                    default="outputs/strict_global_idx/summary.json")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    protocol = _load_protocol(_REPO_ROOT / "configs/protocol.yaml")
    from accpre.models.drafter_mdlm import MDLMDrafter
    from accpre.models.verifier_gpt2 import GPT2Verifier
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {
        "float32": torch.float32, "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[protocol.dtype]
    drafter = MDLMDrafter(
        model_name=protocol.drafter_model, device=device, dtype=dtype,
    )
    drafter.model.eval()
    verifier = GPT2Verifier(
        model_name=protocol.verifier_model, device=device, dtype=dtype,
    )

    pool = test_split()[:int(args.n_prompts)]
    prompt_indices = list(
        range(TRAIN_N + VAL_N, TRAIN_N + VAL_N + int(args.n_prompts))
    )

    # Run oracle-exec for each τ.
    results: Dict = {"per_tau": {}}
    for tau in [float(x) for x in args.taus.split(",") if x.strip()]:
        r = run_oracle_lane(
            drafter, verifier, pool, prompt_indices, protocol,
            gamma=8, T=2, max_new_tokens=int(args.max_new_tokens), tau=tau,
        )
        delta_nll = r["nll_mean"] - STRICT_REF_NLL
        r["delta_NLL_vs_strict"] = delta_nll
        results["per_tau"][str(tau)] = r
        print(
            f"[oracle_exec] τ={tau}  tok/s={r['tok_s_mean']:.2f}  "
            f"NLL={r['nll_mean']:.4f}  ΔNLL={delta_nll:+.4f}  "
            f"n_scored={r['n_scored_total']}"
        )

    with open(out_dir / "oracle_exec.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[oracle_exec] wrote {out_dir}/oracle_exec.json")

    # Load other rows for combined pareto + summary table.
    with open(_REPO_ROOT / args.phase20_dir / "system_table.json") as f:
        nll_data = json.load(f)
    with open(_REPO_ROOT / args.oracle_json) as f:
        oracle_off = json.load(f)["full_test_split"]
    strict_tok_s = 19.21
    if (_REPO_ROOT / args.strict_json).is_file():
        with open(_REPO_ROOT / args.strict_json) as f:
            strict_tok_s = float(json.load(f)["strict"]["tok_s_mean"])

    points: List[Dict] = [
        {"method": "strict", "tau": "—",
         "tok_s": strict_tok_s, "delta_NLL": 0.0, "NLL": STRICT_REF_NLL},
    ]
    for method in ("frozen_1A", "live_jnt_5ep"):
        for tau in (0.5, 0.7, 0.9):
            m = nll_data["methods"].get(method, {}).get(str(tau))
            if m is None:
                continue
            points.append({
                "method": method, "tau": float(tau),
                "tok_s": m["tok_s_mean"], "delta_NLL": m["delta_nll_vs_strict"],
                "NLL": m["nll_mean"],
            })
    for tau in (0.5, 0.7, 0.9):
        r = results["per_tau"].get(str(tau))
        if r is None:
            continue
        points.append({
            "method": "oracle_exec", "tau": float(tau),
            "tok_s": r["tok_s_mean"], "delta_NLL": r["delta_NLL_vs_strict"],
            "NLL": r["nll_mean"],
        })

    _pareto_figure(
        points, out_dir / "pareto_with_oracle.png",
        strict_point=(0.0, strict_tok_s),
    )

    # Summary table.
    lines = [
        "# Phase 20 — executable oracle-Q2 + Pareto update\n\n",
        "```\n",
        f"  {'method':<14}{'τ':>6}{'offCF1':>8}{'tok/s':>8}{'NLL':>10}{'ΔNLL':>10}\n",
    ]
    # strict
    lines.append(
        f"  {'strict':<14}{'—':>6}{'1.0000':>8}{strict_tok_s:>8.2f}"
        f"{STRICT_REF_NLL:>10.4f}{'0.0000':>10}\n"
    )
    # oracle_exec rows, with offCF1 from stored records (decision ceiling).
    for tau in (0.5, 0.7, 0.9):
        r = results["per_tau"].get(str(tau))
        if r is None:
            continue
        off = oracle_off.get(str(tau), {})
        lines.append(
            f"  {'oracle_exec':<14}{float(tau):>6.1f}"
            f"{off.get('cf_at_1', float('nan')):>8.4f}"
            f"{r['tok_s_mean']:>8.2f}"
            f"{r['nll_mean']:>10.4f}"
            f"{r['delta_NLL_vs_strict']:>10.4f}\n"
        )
    # comparison rows from phase20
    for method in ("frozen_1A", "live_jnt_5ep"):
        for tau in (0.5, 0.7, 0.9):
            m = nll_data["methods"].get(method, {}).get(str(tau))
            if m is None:
                continue
            lines.append(
                f"  {method:<14}{float(tau):>6.1f}"
                f"{'  —   ':>8}"      # offline CF@1 shown in earlier tables
                f"{m['tok_s_mean']:>8.2f}"
                f"{m['nll_mean']:>10.4f}"
                f"{m['delta_nll_vs_strict']:>10.4f}\n"
            )
    lines.append("```\n")
    report = "".join(lines)
    with open(out_dir / "oracle_exec_summary.md", "w") as f:
        f.write(report)
    print(report)
    print(f"[oracle_exec] wrote {out_dir}/oracle_exec_summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())

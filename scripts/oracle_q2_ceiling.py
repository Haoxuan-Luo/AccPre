"""Task A — oracle-Q2 CF ceiling under the current commit_threshold rule.

Read-only. For each test record and each τ in {0.5, 0.7, 0.9}:
    L_oracle  =  commit_threshold(record.min_pq_j, τ)
    compare   to  record.L  (strict-rule ground truth)

Reports CF@1 (exact), under, over, and the induced L_hat distribution,
both on the full test split (gamma=8) and on the online-style subset
(prompt indices 60..69, matching the 10-prompt × 64-token online runs).

This is not a predictor. It is the *decision ceiling* of the canonical
commit_threshold rule when given oracle Q2. If the oracle ceiling itself
is low at a given τ, no acceptance predictor can beat it at that τ under
this commit rule.

Outputs:
    outputs/oracle_q2_ceiling/report.md
    outputs/oracle_q2_ceiling/summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from accpre.core.commit import commit_threshold
from accpre.core.protocol import ProtocolConfig
from accpre.core.schema import load_records
from accpre.data.splits import POOL_SIZE, TRAIN_N, VAL_N


def load_protocol(path: Path) -> ProtocolConfig:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    for k in ("draft_salt", "accept_salt", "fallback_salt",
              "schema_version", "max_verifier_ctx"):
        if k in d and isinstance(d[k], str):
            d[k] = int(d[k], 0)
    return ProtocolConfig(**d)


def oracle_per_tau(records, tau: float) -> Dict:
    """Compute L_oracle, compare to record.L. Aggregate over records."""
    n = len(records)
    exact = under = over = 0
    L_hat_counter: Counter = Counter()
    L_strict_counter: Counter = Counter()
    abs_diffs: List[int] = []
    per_prompt: Dict[int, List[int]] = {}
    for r in records:
        L_hat = commit_threshold(r.min_pq_j, float(tau))
        L_hat = max(0, min(L_hat, r.gamma))
        L_strict = int(r.L)
        L_hat_counter[int(L_hat)] += 1
        L_strict_counter[L_strict] += 1
        abs_diffs.append(int(abs(L_hat - L_strict)))
        if L_hat == L_strict:
            exact += 1
        elif L_hat < L_strict:
            under += 1
        else:
            over += 1
        per_prompt.setdefault(int(r.prompt_idx), []).append(
            1 if L_hat == L_strict else 0
        )
    per_prompt_mean = {p: sum(v) / len(v) for p, v in per_prompt.items()}
    return {
        "n": int(n),
        "tau": float(tau),
        "cf_at_1": float(exact) / max(n, 1),
        "under": float(under) / max(n, 1),
        "over": float(over) / max(n, 1),
        "mean_abs_diff": (
            sum(abs_diffs) / max(len(abs_diffs), 1)
        ),
        "L_hat_distribution": {str(k): int(L_hat_counter[k])
                               for k in sorted(L_hat_counter)},
        "L_strict_distribution": {str(k): int(L_strict_counter[k])
                                  for k in sorted(L_strict_counter)},
        "per_prompt_cf_mean": per_prompt_mean,
        "n_prompts": len(per_prompt_mean),
    }


def format_table(rows_by_tau: Dict[float, Dict], gamma: int) -> str:
    lines: List[str] = []
    taus = sorted(rows_by_tau.keys())
    hdr = f"{'metric':<24}" + "".join(f"{'τ='+str(t):>10}" for t in taus) + "\n"
    lines.append(hdr)
    for field, label in [
        ("cf_at_1", "CF@1 (exact)"),
        ("under",   "under rate"),
        ("over",    "over rate"),
        ("mean_abs_diff", "mean |L̂ - L_strict|"),
    ]:
        row = f"{label:<24}"
        for t in taus:
            v = rows_by_tau[t][field]
            row += f"{v:>10.4f}"
        lines.append(row + "\n")
    lines.append("\n")
    # L_hat distribution table.
    lines.append(f"{'L̂ distribution (fraction)':<24}" +
                 "".join(f"{'τ='+str(t):>10}" for t in taus) + "\n")
    for k in range(0, gamma + 1):
        row = f"{'  L̂ == '+str(k):<24}"
        for t in taus:
            dist = rows_by_tau[t]["L_hat_distribution"]
            v = int(dist.get(str(k), 0))
            n = int(rows_by_tau[t]["n"])
            row += f"{(v/max(n,1)):>10.4f}"
        lines.append(row + "\n")
    # L_strict reference (same for all τ on the same records).
    first = rows_by_tau[taus[0]]
    lines.append("\nL_strict distribution (reference, same across τ):\n")
    n = int(first["n"])
    for k in range(0, gamma + 1):
        v = int(first["L_strict_distribution"].get(str(k), 0))
        lines.append(f"  L_strict == {k}: {v/max(n,1):.4f}  (n={v})\n")
    return "".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=str, default=str(_REPO_ROOT))
    ap.add_argument("--gamma", type=int, default=8)
    ap.add_argument("--out_dir", type=str,
                    default=str(_REPO_ROOT / "outputs/oracle_q2_ceiling"))
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    protocol = load_protocol(root / "configs/protocol.yaml")
    all_recs = load_records(
        str(root / "data_collected/stage1_pp.pt"), expected_protocol=protocol,
    )
    gamma = int(args.gamma)
    gamma_filtered = [r for r in all_recs if r.gamma == gamma]

    # Full test split (prompts 60..79).
    test_prompts_full = set(range(TRAIN_N + VAL_N, POOL_SIZE))
    test_records_full = [
        r for r in gamma_filtered if r.prompt_idx in test_prompts_full
    ]

    # Online-subset (10 prompts × 64 tokens, prompts 60..69). Same records
    # as filtered to the first 10 test prompts.
    online_subset_prompts = set(range(TRAIN_N + VAL_N, TRAIN_N + VAL_N + 10))
    test_records_online = [
        r for r in gamma_filtered if r.prompt_idx in online_subset_prompts
    ]

    print(
        f"[oracle_q2] full test  : n={len(test_records_full)}  "
        f"prompts={sorted(test_prompts_full)}"
    )
    print(
        f"[oracle_q2] online sub.: n={len(test_records_online)}  "
        f"prompts={sorted(online_subset_prompts)}"
    )

    taus = [0.5, 0.7, 0.9]
    full_by_tau = {t: oracle_per_tau(test_records_full, t) for t in taus}
    online_by_tau = {t: oracle_per_tau(test_records_online, t) for t in taus}

    summary = {
        "protocol_fingerprint": protocol.fingerprint(),
        "gamma": gamma,
        "n_full_test": len(test_records_full),
        "n_online_subset": len(test_records_online),
        "full_test_split": full_by_tau,
        "online_subset": online_by_tau,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    report: List[str] = []
    report.append("# Task A — oracle-Q2 CF ceiling under commit_threshold\n\n")
    report.append(
        f"Protocol fingerprint: `{protocol.fingerprint()[:80]}...`\n\n"
    )
    report.append(
        f"Gamma: {gamma}.  Full test: n={len(test_records_full)} across "
        f"{len(test_prompts_full)} prompts.  Online subset: "
        f"n={len(test_records_online)} across "
        f"{len(online_subset_prompts)} prompts.\n\n"
    )
    report.append("Reading: `L_oracle = commit_threshold(record.min_pq_j, τ)`, "
                  "compare to strict `record.L`. "
                  "Under/over are signed: `under` ⟺ `L_oracle < L_strict` "
                  "(predictor rejects faster than strict); "
                  "`over` ⟺ `L_oracle > L_strict`.\n\n")
    report.append("## Full test split\n\n```\n")
    report.append(format_table(full_by_tau, gamma))
    report.append("```\n\n## Online-style subset (prompts 60..69)\n\n```\n")
    report.append(format_table(online_by_tau, gamma))
    report.append("```\n\n")

    with open(out_dir / "report.md", "w") as f:
        f.write("".join(report))

    print("\n".join([
        "",
        "== Full test (oracle-Q2 CF ceiling) ==",
        format_table(full_by_tau, gamma),
        "",
        "== Online subset ==",
        format_table(online_by_tau, gamma),
    ]))
    print(f"\n[oracle_q2] wrote {out_dir}/summary.json")
    print(f"[oracle_q2] wrote {out_dir}/report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())

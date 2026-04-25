"""Phase 9 Step-1 sanity report for acc_tx_v1.

Compares acc_tx_v1 (per-position transformer, family hidden_per_pos_v1)
against pp_base (per-position MLP, family hidden_per_pos) on:

  - val MSE from train_history.json
  - online CF@1 (aggregate) per tau
  - online tok/s mean per tau

Purpose is plumbing sanity. Either the numbers line up within a loose
envelope (indicating the transformer encoder is correctly wired) or
they do not (indicating a bug worth fixing before we expand the
family). We do NOT gate L1 on this.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

# Repo root on sys.path so this script works whether invoked from
# anywhere. Defensive; the current script does not import accpre but
# its siblings (sanity_l1.py) do, so we match their convention.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

TX_CKPT = "checkpoints/acc_tx_v1"
PP_CKPT = "checkpoints/acc_frz_hid_pp"
PP_ONLINE = "outputs/phase3_online_pp_11799951/acc_frz_hid_pp"
TAUS = (0.3, 0.5, 0.7, 0.9)


def _read_best_val(ckpt_dir: str) -> Optional[float]:
    path = os.path.join(ckpt_dir, "train_history.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        return float(json.load(f).get("best_val", float("nan")))


def _read_online(dir_: str, tau: float):
    tag = f"{tau:.1f}".replace(".", "p")
    path = os.path.join(dir_, f"online_acceptance_tau_{tag}.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        j = json.load(f)
    s = j["l3"]
    return {
        "tok_s_mean": float(s["tok_s_mean"]),
        "cf_at_1_aggregate": float(s["cf_at_1_aggregate"]),
    }


def main(tx_online: str) -> int:
    print("=" * 80)
    print("Phase 9 Step-1 sanity: acc_tx_v1  vs  acc_frz_hid_pp (pp_base)")
    print("=" * 80)

    tx_val = _read_best_val(TX_CKPT)
    pp_val = _read_best_val(PP_CKPT)
    print()
    print(f"val MSE   acc_tx_v1  : {tx_val if tx_val is not None else 'MISSING'}")
    print(f"val MSE   pp_base    : {pp_val if pp_val is not None else 'MISSING'}")
    if tx_val is not None and pp_val is not None and pp_val > 0:
        rel = (tx_val - pp_val) / pp_val
        print(f"delta_rel (tx - pp)/pp : {rel:+.4f}")
        print(f"envelope  +/- 0.20     : {'WITHIN' if abs(rel) <= 0.20 else 'OUTSIDE'}")

    print()
    print(f"{'tau':<4}  {'tx_tok_s':>10} {'pp_tok_s':>10} {'tx/pp':>6}   "
          f"{'tx_CF@1':>8} {'pp_CF@1':>8} {'delta':>7}")
    print("-" * 80)
    worst_cf_delta = 0.0
    for t in TAUS:
        tx = _read_online(tx_online, t)
        pp = _read_online(PP_ONLINE, t)
        if tx is None or pp is None:
            print(f"{t:<4}  MISSING  tx={tx is not None}  pp={pp is not None}")
            continue
        tok_ratio = tx["tok_s_mean"] / max(pp["tok_s_mean"], 1e-9)
        cf_delta = tx["cf_at_1_aggregate"] - pp["cf_at_1_aggregate"]
        worst_cf_delta = max(worst_cf_delta, abs(cf_delta))
        print(
            f"{t:<4}  {tx['tok_s_mean']:>10.2f} {pp['tok_s_mean']:>10.2f} "
            f"{tok_ratio:>5.2f}x   {tx['cf_at_1_aggregate']:>8.3f} "
            f"{pp['cf_at_1_aggregate']:>8.3f} {cf_delta:+7.3f}"
        )

    print()
    print("Envelope for the Step-1 sanity (not a gate, documentation only):")
    print("  * val MSE within +/- 20% of pp_base ......... diagnostic")
    print("  * CF@1 at each tau within +/- 0.02 of pp_base  diagnostic")
    print(
        f"  worst CF@1 |delta| across taus = {worst_cf_delta:.3f}  "
        f"({'WITHIN' if worst_cf_delta <= 0.02 else 'OUTSIDE'} the 0.02 band)"
    )
    print()
    print("Remember: the L1 sanity will proceed regardless of this outcome.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python sanity_a1.py <tx_online_dir>", file=sys.stderr)
        print("  <tx_online_dir> should point to the acc_tx_v1 online output folder",
              file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))

"""Phase 23 sanity — live-joint token-id path fix.

Verifies two invariants that together show the fix from

  `_forward_item`  /  `_save_joint_predictions`

(switch the tok-emb input from STORED `batch["token_ids"][i]` to
REPLAYED `draft_tokens` when `live_q2_target=True`) is safe and actually
activates under drafter drift.

Invariant A (no-op at step 0):
  Under an UNMODIFIED pretrained drafter + matched RNG, the replayed
  draft_tokens must equal the stored `record.draft_tokens` bit-for-bit.
  So the new live-path tok-emb input is identical to the old one
  at the start of training. Any live-vs-lazy divergence therefore
  comes ONLY from drafter drift during fine-tuning, not from the
  code change itself.

Invariant B (fix activates under drift):
  Under a PERTURBED drafter (we load a joint-checkpoint drafter.pt —
  typically `checkpoints/acc_jnt_1a_liveq2_long/drafter.pt`), the
  replayed draft_tokens diverge from stored on at least one record.
  And we re-run `_joint_assemble_v2` with BOTH stored tok-ids and
  replayed tok-ids under the 1A head; the q_hat values differ on
  positions where tokens differ, confirming that feeding replayed vs
  stored tok-ids is not a no-op under drift.

Exit 0 on pass, 1 on fail.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from accpre.core.draft_verify import _make_generator
from accpre.core.protocol import ProtocolConfig
from accpre.core.schema import load_records
from accpre.data.splits import POOL_SIZE, TRAIN_N, VAL_N
from accpre.train.cli import _joint_assemble_v2
from accpre.train.dataset import JointAcceptanceDataset


def load_protocol(path: Path) -> ProtocolConfig:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    for k in (
        "draft_salt", "accept_salt", "fallback_salt",
        "schema_version", "max_verifier_ctx",
    ):
        if k in d and isinstance(d[k], str):
            d[k] = int(d[k], 0)
    return ProtocolConfig(**d)


def _pick_records(split_records) -> List[int]:
    picks: List[int] = [0, 1, 2]
    by_prompt: dict = {}
    for pos, r in enumerate(split_records):
        prev = by_prompt.get(int(r.prompt_idx))
        if prev is None or int(r.round_idx) > int(split_records[prev].round_idx):
            by_prompt[int(r.prompt_idx)] = pos
    for pos in sorted(by_prompt.values()):
        picks.append(pos)
    seen = set()
    dedup = []
    for p in picks:
        if p in seen:
            continue
        seen.add(p)
        dedup.append(p)
    return dedup


def _replay_tokens(drafter, prefix, seed, gamma, T, protocol):
    """Replay drafter.draft_with_features with matched RNG, returning
    (draft_tokens, draft_log_probs, drafter_hidden_per_pos) on device."""
    draft_rng = _make_generator(drafter.device, seed ^ protocol.draft_salt)
    draft_tokens, draft_log_probs, drafter_hidden = drafter.draft_with_features(
        prefix_ids=prefix, gamma=int(gamma), T=int(T),
        temperature=protocol.temperature, q_mode=protocol.q_mode,
        generator=draft_rng, pool="per_position", layers=(-1,),
    )
    return draft_tokens, draft_log_probs, drafter_hidden


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=str, default=str(_REPO_ROOT))
    ap.add_argument("--split", type=str, default="test",
                    choices=["train", "val", "test"])
    ap.add_argument(
        "--perturbed_drafter_ckpt", type=str,
        default="checkpoints/acc_jnt_1a_liveq2_long/drafter.pt",
        help="Path to a joint-checkpoint drafter.pt for Invariant B. "
             "Relative to --root.",
    )
    ap.add_argument(
        "--head_ckpt_dir", type=str,
        default="checkpoints/acc_jnt_1a_liveq2_long",
        help="Checkpoint directory holding the 1A head (model.pt + "
             "config.yaml) used to exercise Invariant B. Relative to "
             "--root.",
    )
    ap.add_argument("--n_records", type=int, default=16)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    protocol = load_protocol(root / "configs/protocol.yaml")
    print(f"[tokemb_fix_sanity] protocol fp: {protocol.fingerprint()[:60]}...")

    all_recs = load_records(
        str(root / "data_collected/stage1_pp.pt"),
        expected_protocol=protocol,
    )
    gamma_filtered = [r for r in all_recs if r.gamma == 8]
    ranges = {
        "train": (0, TRAIN_N),
        "val": (TRAIN_N, TRAIN_N + VAL_N),
        "test": (TRAIN_N + VAL_N, POOL_SIZE),
    }
    lo, hi = ranges[args.split]
    split_prompts = set(range(lo, hi))
    split_records = [r for r in gamma_filtered if r.prompt_idx in split_prompts]
    print(
        f"[tokemb_fix_sanity] {args.split} split: n={len(split_records)} "
        f"records across prompts {lo}..{hi - 1}"
    )
    ds = JointAcceptanceDataset(split_records, family="hidden_per_pos_v2")
    picks = _pick_records(split_records)[: args.n_records]
    print(f"[tokemb_fix_sanity] checking {len(picks)} records at positions {picks}")

    from accpre.models.drafter_mdlm import MDLMDrafter
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[protocol.dtype]
    drafter = MDLMDrafter(
        model_name=protocol.drafter_model, device=device, dtype=dtype,
    )
    drafter.model.eval()

    # -----------------------------------------------------------------
    # Invariant A — unmodified drafter: replayed == stored (bit-exact)
    # -----------------------------------------------------------------
    print("\n[tokemb_fix_sanity] --- Invariant A: unmodified drafter ---")
    n_fail_a = 0
    rows_a = []
    for pos in picks:
        r = split_records[pos]
        item = ds[pos]
        prefix = item["prefix_ids"].to(device)
        seed = int(item["round_rng_seed"].item())
        draft_tokens, _, _ = _replay_tokens(
            drafter, prefix, seed, int(r.gamma), int(r.T), protocol,
        )
        stored = [int(t) for t in r.draft_tokens]
        replayed = [int(t) for t in draft_tokens.cpu().tolist()]
        ok = (stored == replayed)
        if not ok:
            n_fail_a += 1
        rows_a.append({
            "pos": int(pos),
            "prompt_idx": int(r.prompt_idx),
            "round_idx": int(r.round_idx),
            "prefix_len": int(prefix.shape[0]),
            "bit_exact_match": bool(ok),
        })
        print(
            f"[tokemb_fix_sanity]   {'OK' if ok else 'FAIL'}  "
            f"prompt={r.prompt_idx:2d} round={r.round_idx:2d} "
            f"prefix_len={int(prefix.shape[0]):3d}  "
            f"bit_exact={ok}"
        )
    if n_fail_a > 0:
        print(f"[tokemb_fix_sanity] Invariant A FAIL — {n_fail_a}/{len(picks)} mismatched")

    # -----------------------------------------------------------------
    # Invariant B — perturbed drafter: replayed ≠ stored on at least one
    # record, and head q_hat changes under stored-vs-replayed tok-ids.
    # -----------------------------------------------------------------
    print("\n[tokemb_fix_sanity] --- Invariant B: perturbed drafter ---")
    pert_path = root / args.perturbed_drafter_ckpt
    head_cfg_path = root / args.head_ckpt_dir / "config.yaml"
    head_pt_path = root / args.head_ckpt_dir / "model.pt"
    if not pert_path.exists():
        print(
            f"[tokemb_fix_sanity] (skip B) perturbed drafter ckpt missing: "
            f"{pert_path}"
        )
        n_fail_b = 0
        invariant_b_ran = False
        n_drift_records = 0
        n_q_hat_diff_records = 0
        rows_b: list = []
    elif not head_cfg_path.exists() or not head_pt_path.exists():
        print(
            f"[tokemb_fix_sanity] (skip B) head ckpt missing under "
            f"{head_cfg_path.parent}"
        )
        n_fail_b = 0
        invariant_b_ran = False
        n_drift_records = 0
        n_q_hat_diff_records = 0
        rows_b = []
    else:
        invariant_b_ran = True
        print(f"[tokemb_fix_sanity]   loading perturbed drafter from {pert_path}")
        drafter.model.load_state_dict(
            torch.load(str(pert_path), map_location=device)
        )
        drafter.model.eval()

        # Build the 1A head on CPU and move to device.
        with open(head_cfg_path) as f:
            head_cfg = yaml.safe_load(f)
        from accpre.predictors.pp_tokemb import AcceptanceMLPPerPosTokEmb
        head = AcceptanceMLPPerPosTokEmb(
            family=str(head_cfg["family"]),
            gamma=int(head_cfg.get("gamma", 8)),
            hidden_dim=int(head_cfg.get("hidden_dim", 128)),
            dropout=float(head_cfg.get("dropout", 0.1)),
            token_emb_dim=int(head_cfg.get("token_emb_dim", 64)),
        ).to(device)
        head.load_state_dict(torch.load(str(head_pt_path), map_location=device))
        head.eval()

        n_drift_records = 0
        n_q_hat_diff_records = 0
        rows_b = []
        for pos in picks:
            r = split_records[pos]
            item = ds[pos]
            prefix = item["prefix_ids"].to(device)
            seed = int(item["round_rng_seed"].item())
            draft_tokens, draft_log_probs, drafter_hidden = _replay_tokens(
                drafter, prefix, seed, int(r.gamma), int(r.T), protocol,
            )
            stored = torch.tensor(
                [int(t) for t in r.draft_tokens],
                dtype=torch.long, device=device,
            )
            replayed = draft_tokens.long().to(device)
            tokens_drifted = bool((stored != replayed).any().item())
            if tokens_drifted:
                n_drift_records += 1

            # Build v2 feats from the REPLAYED drafter outputs (same as
            # training / _forward_item).
            feats = _joint_assemble_v2(
                drafter_hidden.to(device),
                draft_log_probs.to(device),
                replayed,
            )
            with torch.no_grad():
                q_hat_replayed = head.forward(
                    feats.unsqueeze(0),
                    token_ids=replayed.unsqueeze(0),
                ).squeeze(0).detach().cpu()
                q_hat_stored = head.forward(
                    feats.unsqueeze(0),
                    token_ids=stored.unsqueeze(0),
                ).squeeze(0).detach().cpu()
            q_hat_abs_max = float((q_hat_replayed - q_hat_stored).abs().max().item())
            q_hat_diff = q_hat_abs_max > 1e-6
            if q_hat_diff:
                n_q_hat_diff_records += 1

            rows_b.append({
                "pos": int(pos),
                "prompt_idx": int(r.prompt_idx),
                "round_idx": int(r.round_idx),
                "prefix_len": int(prefix.shape[0]),
                "tokens_drifted": bool(tokens_drifted),
                "q_hat_abs_max_diff": q_hat_abs_max,
                "q_hat_diff_nontrivial": bool(q_hat_diff),
            })
            print(
                f"[tokemb_fix_sanity]   prompt={r.prompt_idx:2d} "
                f"round={r.round_idx:2d} "
                f"tokens_drifted={tokens_drifted}  "
                f"|q_hat_replayed - q_hat_stored|_inf={q_hat_abs_max:.2e}"
            )

        # Fail only if ALL records agree on tokens — then the perturbed
        # drafter isn't actually perturbed relative to collection (and
        # Invariant B cannot be exercised with the provided ckpt).
        if n_drift_records == 0:
            print(
                "[tokemb_fix_sanity] Invariant B indeterminate: the "
                "perturbed drafter did not produce any token drift on "
                "the checked records. Point --perturbed_drafter_ckpt at "
                "a checkpoint that actually differs from the pretrained "
                "MDLM."
            )
            n_fail_b = 1
        elif n_q_hat_diff_records == 0:
            print(
                "[tokemb_fix_sanity] Invariant B FAIL: tokens drifted "
                "on some records but head q_hat did not change when "
                "swapping stored vs replayed tok-ids — the head is not "
                "actually using the tok-emb input."
            )
            n_fail_b = 1
        else:
            n_fail_b = 0

    summary = {
        "split": args.split,
        "n_checked": len(picks),
        "invariant_A": {
            "n_fail": n_fail_a,
            "rows": rows_a,
        },
        "invariant_B_ran": invariant_b_ran,
        "invariant_B": {
            "n_drift_records": n_drift_records,
            "n_q_hat_diff_records": n_q_hat_diff_records,
            "n_fail": n_fail_b,
            "rows": rows_b,
        } if invariant_b_ran else None,
    }
    out_path = root / "outputs/sanity_joint_tokemb_fix.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[tokemb_fix_sanity] wrote {out_path}")

    n_fail_total = n_fail_a + (n_fail_b if invariant_b_ran else 0)
    if n_fail_total == 0:
        print("[tokemb_fix_sanity] PASS")
        return 0
    print(f"[tokemb_fix_sanity] FAIL  total failures: {n_fail_total}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

# AccPre — Phase 2 Staged Design (predictors)

**Status:** Planning only. No predictor code yet.
**Scope:** Adds predictor *method families* (acceptance + committed-length)
on top of the clean Phase 1 baseline + eval pipeline. Inherits every
Frozen v1 Protocol Decision from `DESIGN.md`: `temperature = 1.0`,
`q_mode = "A"`, wall-clock throughput, CF@1 as primary quality axis,
XL-audit auxiliary only, oracle-Q2 threshold as strong reference.

This document is authoritative for Phase 2 decisions that are not already
locked in `DESIGN.md`. Anything that conflicts with `DESIGN.md` is a bug
in this document.

**Two-view evaluation (primary methodological decision).** Every Phase 2
predictor is evaluated under two independent views; neither is an
approximation of the other, and the final report carries both.

| View | What it measures | How |
|---|---|---|
| **Offline oracle replay** | How well the predictor learns its canonical target | Replay the predictor on stored Phase-1 strict records. Canonical inputs are the logged features; canonical targets (`Q2_j`, `L_τ^oracle`) come from the same records. Emits L1 predictor-level metrics and L2 decision-level metrics. |
| **Online predictor decode** | Real speed-quality tradeoff | Run the predictor in an actual decode loop that **skips the full verifier-on-draft pass** and uses the predictor to decide commits. Emits L3 system-level metrics: wall-clock tok/s and CF@1. |

The main lane is the cheap online decode path: the predictor replaces
strict's per-position `draft-vs-verifier` comparison so the verifier's
γ-token suffix pass is *not* run. Stronger upper-bound lanes that use
richer features (e.g. verifier-prefix hidden) are reported separately
and never land on the main Pareto.

---

## A. Stage breakdown

Phase 2 is split into a frozen-first sequence (`2A`) and a joint /
multitask sequence (`2B`). Within each, offline training is followed
immediately by an **online decode validation** sub-stage so the speed
axis of the Pareto is exercised before later stages pile on complexity.
No stage begins before its predecessor is validated end-to-end.

| Stage | Name | What it does | Prerequisite |
|---|---|---|---|
| **2A.0** | Stage-1 collection | Extend Phase 1 `RoundRecord` with optional feature fields (drafter hidden, verifier prefix hidden + numerics). Re-run Stage 1 to produce `stage1.pt`. No new logic beyond hidden-state extraction. | Phase 1 green |
| **2A.1** | Frozen acceptance predictors (offline) | Train Q2 predictors with a **frozen** backbone / no-backbone, head-only. 3 input families (see §C). Offline L1 + L2 reported. | 2A.0 |
| **2A.2** | Frozen committed-length predictors (offline) | Train τ-conditioned L̂(τ) predictors, head-only. 2 input families. Offline L1 + L2 reported. | 2A.0, 2A.1 |
| **2A.3** | **Online predictor decode (frozen)** | Run every 2A.1/2A.2 predictor in a **real decode loop** that skips the full verifier-on-draft pass and commits via the predictor. First L3 measurements: wall-clock tok/s + CF@1 on the main cheap lane. Only predictors whose input family is compatible with the target deployment mode (§C.3) are eligible. | 2A.1, 2A.2 |
| **— validation gate —** | | All L1 + L2 + L3 reports produced and sanity-checked. Any predictor whose L3 falls outside the strict / oracle-Q2 envelope is investigated before 2B begins. | 2A.3 |
| **2B.1** | Joint single-task (offline + online) | Unfreeze MDLM backbone; fine-tune backbone + one head. Offline L1/L2 as in 2A.1/2A.2. Online L3 in the same cheap-decode loop as 2A.3. | 2A gate |
| **2B.2** | Joint multitask, fixed weights (offline + online) | Shared backbone, both heads, `w_acc·L_acc + w_len·L_len` with fixed weights. Offline + online evaluation as above. | 2B.1 |
| **2B.3** | Joint multitask, learnable weights (offline + online) | 2B.2 plus homoscedastic-uncertainty task weights (Kendall & Gal): `log σ_k` learnable per task. | 2B.2 |

Optional **ablation / diagnostic lanes** (not mandatory to ship):
upper-bound-feature lanes (adds verifier-prefix hidden state); q_mode
"B" ablation; **Option-X diagnostic lane** (full verifier, predictor
replaces commit only — see §H.3). Option X is not the main online lane
in v1; it is a diagnostic for separating "commit-rule quality" from
"missing-fallback cost."

---

## B. Canonical targets

### B.1 Acceptance predictor — primary target: `Q2_j`

```
Q2_j := min(1, p_j / q_j)   (already logged in every Phase 1 RoundRecord)
```

**Why `Q2_j` and not `accepted_j`:**

1. **Regression on the clean signal beats classification on the noisy
   sample.** At `temperature = 1.0`, `accepted_j` is a *Bernoulli sample*
   from `Bernoulli(Q2_j)`. For every logged `(prompt, round, j)`, we see
   exactly one sample — the underlying probability `Q2_j` is a strictly
   better supervised target than its realisation.
2. **Same space as the commit rule.** The oracle-Q2 baseline *is* defined
   by thresholding `Q2_j`. If a predictor produces `Q̂_j ≈ Q2_j`, it
   plugs directly into `commit_threshold(Q̂_j, τ)` — no extra
   calibration layer required.
3. **Per-position well-defined conditional on survival.** Masking by
   `survived_j == 1` handles the "we never reached this position" case
   cleanly; we already log `survived_j`, so no new bookkeeping.
4. **Data efficiency.** One round gives `γ` soft targets for Q2,
   compared to `Σ survived_j` noisy binary labels for `accepted_j`.
   Soft targets make small datasets go further.

**Auxiliary secondary target: `accepted_j`** — retained as an optional
BCE loss a user can opt into by configuration, never the primary. Used
only for diagnostics (e.g., AUROC linking predicted probability to the
sampled label) and for predictors whose architecture naturally outputs
a Bernoulli head.

### B.2 Committed-length predictor — primary target: `L_τ^oracle`

```
L_τ^oracle(record) := commit_threshold(record.min_pq_j, τ)
                     ∈ {0, 1, ..., γ}
```

**Formulation:** τ-conditioned. The predictor takes `(features, τ)` and
outputs a distribution over `{0, 1, ..., γ}`. Training sweeps τ from a
fixed sampling distribution; inference queries any τ the Pareto needs.

**Why `L_τ^oracle` (τ-conditioned) and not `L_strict`:**

1. **One model ↔ the full Pareto.** The primary v1 question is
   "quality-vs-throughput along τ." A τ-conditioned L̂(τ) can produce a
   curve with a single trained model; a τ-unaware model cannot.
2. **Matches the headline commit rule.** The oracle-Q2 baseline commits
   via `commit_threshold(Q2_j, τ)`; the length predictor's target is
   literally the label produced by that rule. So the predictor is a
   *direct emulator* of the oracle-Q2 commit function from cheap
   features.
3. **`L_strict` is one noisy point on the τ-curve.** Under the v1
   Leviathan protocol, strict's commit length is a stochastic function
   of the same inputs; training against it adds label noise without
   adding signal beyond what `L_τ^oracle` already carries.
4. **Clean categorical loss.** `L_τ^oracle ∈ {0, ..., γ}` → `γ+1` classes
   → standard cross-entropy. No ordinal hacks, no custom losses.

**Training-time τ sampling:** uniform on `{0.1, 0.3, 0.5, 0.7, 0.9}` per
example. Endpoints (τ near 0 or 1) are degenerate (`L_τ = γ` or `L_τ =
0` for nearly all rounds) and provide little supervisory signal. The
five-point grid keeps the Pareto-relevant region dense and aligns with
the evaluation τ-sweep.

**Secondary targets, diagnostic only:**
- `L_strict`: a sanity sanity check — we want `argmax P(L|τ*)` at some
  calibrated τ* to be close to `L_strict`.

---

## C. Input families

Every feature below is **per round** and is populated at Stage-1 time
(2A.0), stored in the optional fields of `RoundRecord`. No feature
requires running the drafter or verifier at predictor training or
evaluation time — feature extraction is amortized into Stage-1 once.

### C.1 Main cheap lane (the headline predictor story)

| Family | Feature vector | Online deployment mode | Cost saved vs strict |
|---|---|---|---|
| **numeric-only** | `(verifier_entropy, verifier_margin, verifier_top1_prob)` at last prefix token, i.e. 3 scalars. | **V-prefix**: one verifier forward pass over `prefix_len` tokens per round (required to compute the features). | Saves the γ-token suffix of the verifier pass. |
| **drafter-hidden-only** | mean-pooled last-block MDLM hidden over the γ draft positions. Dim ≈ 768. | **V-free**: no verifier call at inference. | Saves the full `prefix_len + γ` verifier pass. |
| **drafter-hidden + numeric** | concatenation of the two above. | **V-prefix**: same prefix pass as numeric-only. | Saves the γ-token suffix. |

"Cheap" means **no verifier forward pass over the draft positions**,
which is where strict's throughput cost lives. Only
`drafter-hidden-only` is truly **V-free**. The other two families
require one prefix-only verifier pass per round; they still skip the
γ-token suffix that strict must compute.

### C.2 Optional upper-bound lane (reference only)

| Family | Feature vector | Online deployment mode | Notes |
|---|---|---|---|
| **verifier-prefix-hidden + drafter-hidden + numeric** | 768-dim GPT-2 XL last-layer hidden state at last prefix token, plus the above. | **V-prefix**: same prefix pass as numeric-only (the last-layer hidden state is produced by the same forward). | Establishes "what cheap features could achieve with a richer prefix summary." |

Not on the main Pareto; reported in a separate panel as a *ceiling*
reference. Never the headline number.

### C.3 Online deployment modes (authoritative)

The two modes below are the only permitted inference paths for Phase 2.
Each predictor is tagged with one mode; the online decode runner
(`accpre/eval/online_decode.py`, created in Stage 2A.3) dispatches on
the mode.

- **V-free** — *verifier-free*. No verifier forward pass in the decode
  loop after model load. The drafter's forward (already needed) produces
  features directly. Eligible: `drafter-hidden-only`.
- **V-prefix** — *verifier-on-prefix*. One verifier forward over the
  *prefix only* per round (no draft positions). Produces numeric
  features and, if needed, the verifier-prefix hidden state. Eligible:
  `numeric-only`, `drafter-hidden + numeric`, upper-bound family.

Modes never change across rounds within a run. A predictor that trained
on `drafter-hidden + numeric` must deploy V-prefix; it cannot silently
fall back to V-free.

### C.4 Explicitly excluded from Phase 2

- Verifier hidden states over **draft positions** (would require the
  full verifier pass that oracle-Q2 uses — that lane IS the oracle
  baseline, not a predictor).
- Cross-attention between drafter and verifier hidden states.
- Any feature requiring the full verifier-on-draft forward.

### C.5 Online decode path (authoritative)

This is what the online decode runner does each round. All Phase 2
online L3 measurements use exactly this loop; no lane uses a different
inference path.

**Commit convention for cheap online decode (v1):**
- Commit exactly `max(1, L̂)` draft tokens per round (from the already
  computed `draft_tokens`). No separate bonus / fallback token sampling.
- `L̂ = 0` ⇒ commit `draft_tokens[0]` only (guarantees progress of ≥ 1
  token per round).
- `L̂ ≥ 1` ⇒ commit `draft_tokens[:L̂]`.

Rationale: strict's "+1" token comes from the verifier's already-computed
distribution at the rejection position — a free gift from the Leviathan
protocol. Cheap decode skips the verifier on draft, so there is no such
distribution available without paying the cost we were trying to save.
We therefore do not emit an extra token. The accounting is: strict
commits `L_strict + 1` per round; cheap commits `max(1, L̂)` per round.
Both tok/s values are honest.

**Per-round decode loop (pseudocode, applies to every Phase 2 predictor):**

```
running ← prefix_ids
round_idx ← 0
while running_len − prefix_len < max_new_tokens:
    seed ← derive_seed(protocol, prompt_idx, round_idx)
    draft_rng, accept_rng, fallback_rng ← seeds(seed, protocol)
        # accept_rng and fallback_rng are logged for auditability but
        # are NOT consumed by the cheap lane's commit / extra-token path

    # 1. drafter (same as Phase 1)
    draft_tokens, draft_log_probs, drafter_hidden ←
        drafter.draft_with_features(running, γ, T, protocol, draft_rng)

    # 2. mode-specific feature extraction
    if predictor.mode == "V-free":
        features ← build_features_vfree(drafter_hidden)
    elif predictor.mode == "V-prefix":
        # ONE verifier forward over the prefix ONLY
        prefix_hidden, numeric ← verifier.prefix_features(running)
        features ← build_features_vprefix(drafter_hidden, prefix_hidden, numeric)

    # 3. predictor inference
    if predictor.kind == "acceptance":     # outputs Q̂_j
        Q̂ ← predictor.predict_q2(features)
        L̂ ← commit_threshold(Q̂, τ)
    elif predictor.kind == "committed_length":  # outputs L̂
        L̂ ← predictor.predict_L(features, τ)
    L̂ ← clip(L̂, 0, γ)

    # 4. commit
    n_commit ← max(1, L̂)
    running ← concat(running, draft_tokens[:n_commit])
    round_idx += 1
return running
```

The decode loop touches the verifier **zero times** in V-free mode and
**exactly once per round** (prefix-only forward) in V-prefix mode. It
NEVER touches the γ-token suffix of the verifier pass. Timer wraps the
entire loop — same timer discipline as Phase 1.

**Acceptance predictor online path** — emits `Q̂ ∈ [0,1]^γ`, thresholds
with `τ` via `commit_threshold(Q̂, τ)` (the canonical Phase 1 rule)
to get `L̂`. Sweeping τ gives a Pareto curve with one trained
predictor.

**Committed-length predictor online path** — emits categorical
`P(L | features, τ)` over `{0, ..., γ}`; `L̂ = argmax P(L|·)`.
τ is passed as an input; sweeping τ likewise traces a curve with one
trained predictor.

Both predictors use identical surrounding plumbing (draft call, feature
extraction, `max(1, L̂)` commit). The only difference is whether the
predictor outputs `Q̂` (thresholded downstream) or `L̂` (direct).

---

## D. Training modes

| Mode | Backbone | Head | Stage | Memory |
|---|---|---|---|---|
| **frozen** | no grad, features pre-computed | trainable | 2A | minimal |
| **joint (single-task)** | grad through one task head | trainable | 2B.1 | MDLM-sized |
| **joint multitask** | grad through both heads | both trainable | 2B.2 / 2B.3 | MDLM-sized |

**Frozen mode detail (Stage 2A):** Features come from `stage1.pt`
(Stage 2A.0 output). The trainer is a feature-only loader — never loads
the MDLM or the verifier. Training a full epoch on the frozen lane
finishes in seconds on CPU. This is a deliberate choice; the point of
Stage 2A is to iterate fast and cheap.

**Joint mode detail (Stage 2B.*):** Gradients flow through the MDLM
during its final DDPM step (only — intermediate DDPM steps remain
stochastic no-grad, matching inference). The joint trainer calls the
SAME `_denoise_step` helper the inference-time drafter uses; the
Phase-4 anti-drift test (`tests/test_train_infer_consistency.py`,
deferred from Phase 1) fires here for the first time. Joint runs
require GPU.

---

## E. Losses

### E.1 Stage 2A (simple)

| Predictor | Target | Loss | Notes |
|---|---|---|---|
| acceptance | `Q2_j` | **MSE**: `mean_{j: survived_j==1}(Q̂_j − Q2_j)²` | MSE, not MAE, because L2 is differentiable everywhere. Reported metric is MAE (§H). |
| committed-length | `L_τ^oracle` | **Cross-entropy** over `{0, ..., γ}`. | Standard categorical. Optionally label-smoothing ε=0.05, but default off. |

Auxiliary / off by default:
- BCE on `accepted_j` masked by `survived_j` — available as a config
  flag for acceptance predictors; never the primary objective.
- Ordinal loss for length — out of scope; revisit only if flat CE
  overcommits at tails.

### E.2 Stage 2B (multitask)

- **2B.1 (single-task joint):** Same individual loss as 2A, but
  gradients flow through the MDLM backbone.
- **2B.2 (multitask, fixed weights):**
  ```
  L_total = w_acc · L_acc + w_len · L_len
  ```
  Fixed `w_acc = w_len = 1.0` on mean-scale-normalised losses
  (empirical scaling documented in the run's config; no learned
  weights yet).
- **2B.3 (multitask, learnable weights):** homoscedastic uncertainty
  weighting (Kendall & Gal 2018): learnable `log σ_k` per task,
  ```
  L_total = Σ_k [ (1 / (2 σ_k²)) · L_k + log σ_k ]
  ```
  One scalar parameter per task. Standard implementation. No further
  tuning knobs.

Explicitly out of scope in v1: ranking-aware losses, margin losses,
threshold-aware losses, focal/hinge variants, task-annealing schedules.
These may be Phase 3+ work.

---

## F. Hyperparameter philosophy

One canonical default config. Fixed for Stage 2A. Stage 2B may override
(smaller batch, lower LR) but must document the override in its config.

| Knob | Default | Scope | Rationale |
|---|---|---|---|
| optimizer | AdamW | all | Standard. |
| batch size | 128 (2A), 16 (2B) | size-dependent | Frozen heads are tiny; 128 is comfortable on CPU. 2B has MDLM in memory. |
| backbone LR | 5e-5 | 2B only | Standard transformer fine-tune. |
| head LR | 1e-3 | all | Standard small-MLP head. |
| weight decay | 1e-4 | all | Standard. |
| warmup | linear, 5% of total steps | all | Light warmup. |
| max epochs | 10 | all | Datasets are small (O(10⁴) rounds). Should converge well inside 10. |
| early stopping | no-improvement for 2 epochs on val loss, restore best | all | Standard. |
| seed | 42 (train/val split), 0 (torch) | all | Fixed; splits come from `accpre.data.splits`. |
| dropout | 0.1 | heads | Standard. |

**Experimental variables for Phase 2 (the only things that move):**
1. Target (acceptance vs length).
2. Input family (numeric / drafter-hidden / both / +upper-bound).
3. Frozen vs joint.
4. Single-task vs multitask vs learnable-weight multitask.

Everything else stays pinned. **No hyperparameter sweeps in Phase 2.**
A sweep is a reasonable Phase 3 follow-up once we know which axes
matter.

---

## G. First experiment matrix

Compact and staged. Stage 2A covers the main frozen story; Stage 2B is
added only after 2A validates.

### Stage 2A matrix (must all run)

Offline training + offline L1/L2 + online L3. "Deploy mode" is the
inference mode of the online decode lane (§C.3).

| ID | Stage | Target | Family | Train mode | Deploy mode |
|---|---|---|---|---|---|
| `acc_frz_num` | 2A.1 → 2A.3 | Q2 | numeric-only | frozen | V-prefix |
| `acc_frz_hid` | 2A.1 → 2A.3 | Q2 | drafter-hidden-only | frozen | **V-free** |
| `acc_frz_hidnum` | 2A.1 → 2A.3 | Q2 | drafter-hidden + numeric | frozen | V-prefix |
| `len_frz_num` | 2A.2 → 2A.3 | L_τ | numeric-only | frozen | V-prefix |
| `len_frz_hidnum` | 2A.2 → 2A.3 | L_τ | drafter-hidden + numeric | frozen | V-prefix |

5 predictors trained and run online. The `acc_frz_hid` entry is the
only one eligible for V-free decode; it is the fastest lane in
expectation and carries the primary speed-quality question for Phase 2.

### Stage 2B matrix (conditional on 2A validating)

Start from the best Stage 2A input family (likely `hidnum` based on the
old project's observations — to be confirmed by 2A results). Run:

| ID | Stage | Target | Family | Train mode | Deploy mode |
|---|---|---|---|---|---|
| `acc_jnt_hid` | 2B.1 | Q2 | drafter-hidden-only | joint single-task | **V-free** |
| `acc_jnt_hidnum` | 2B.1 | Q2 | drafter-hidden + numeric | joint single-task | V-prefix |
| `len_jnt_hidnum` | 2B.1 | L_τ | drafter-hidden + numeric | joint single-task | V-prefix |
| `mt_jnt_hidnum` | 2B.2 | Q2 + L_τ | drafter-hidden + numeric | joint multitask, fixed w | V-prefix |
| `mt_jnt_lw_hidnum` | 2B.3 | Q2 + L_τ | drafter-hidden + numeric | joint multitask, learnable w | V-prefix |

5 predictors trained and run online. `acc_jnt_hid` added to the 2B
matrix so the V-free lane has a joint counterpart to compare against
the frozen version. Total Phase 2 ≈ 10 predictors.

### Optional ablation matrix (only if time permits)

- `acc_frz_upper`: upper-bound lane (adds verifier-prefix hidden). One
  run, informational.
- `len_frz_upper`: same.
- `acc_frz_hidnum_qmodeB`: q_mode "B" ablation — tests whether the
  frozen acceptance lane is sensitive to the q definition. One run.

---

## H. Evaluation plan

Every trained predictor produces a three-block report under two
independent evaluation views. The blocks never share headline numbers.

### H.1 Offline oracle-replay view (L1 + L2)

**Protocol:** replay the predictor on stored Phase-1 strict records.
For each record, feed the logged features through the predictor; compare
predictions against canonical targets (`Q2_j` from the record for
acceptance, `L_τ^oracle = commit_threshold(record.min_pq_j, τ)` for
length). No decode loop is executed; this view measures predictor
learning alone, decoupled from the rest of the decoder.

**L1 — Predictor-level accuracy**

*Acceptance predictors, masked by `survived_j == 1`:*

| Metric | Definition |
|---|---|
| `q2_mae` | `mean |Q̂_j − Q2_j|` — primary "average error." |
| `q2_bias` | `mean (Q̂_j − Q2_j)` — signed. |
| `q2_brier` | `mean (Q̂_j − Q2_j)²`. |
| `q2_ece` | 10-bin calibration error treating `Q̂_j` as a probability. |
| `acc_auc` | AUROC of `Q̂_j` against `accepted_j` — auxiliary link to the binary sample. |

*Length predictors, per τ and aggregated over τ grid:*

| Metric | Definition |
|---|---|
| `len_mae(τ)` | `mean |L̂(τ) − L_τ^oracle|` — primary "average error." |
| `len_bias(τ)` | `mean (L̂(τ) − L_τ^oracle)` — signed. |
| `len_exact(τ)` | Fraction with `L̂(τ) == L_τ^oracle`. |
| `len_over(τ)` | Fraction with `L̂(τ) > L_τ^oracle`. |
| `len_under(τ)` | Fraction with `L̂(τ) < L_τ^oracle`. |
| `len_mae_avg` | Mean of `len_mae(τ)` across the eval τ grid. |

**L2 — Decision-level agreement** (per DESIGN.md §D.4)

For acceptance predictors, threshold via `commit_threshold(Q̂_j, τ)` to
derive `L̂(τ)`; for length predictors `L̂(τ)` is direct. Then compute
against strict's `L_strict` and oracle-Q2's `L_τ^oracle` from the same
record:

| Metric | Definition |
|---|---|
| `L_agreement_strict` | fraction with `L̂(τ) == L_strict` |
| `L_agreement_oracleQ2(τ)` | fraction with `L̂(τ) == L_τ^oracle` — should be high for well-trained Q2 predictors |
| `L_overcommit_rate_strict`, `L_undercommit_rate_strict` | signed diffs vs strict |
| `L_overcommit_token_count` | total tokens committed beyond `L_strict` — quality-violating budget |

`L_agreement_strict` under the v1 shared-fallback coupling is
numerically equal to **offline CF@1**, per DESIGN.md §D.5.2 derivation.
It is reported as `L_agreement_strict` in L2 (not CF@1 under that name),
because CF@1 is reserved for the online view's headline.

### H.2 Online predictor-decode view (L3)

**Protocol:** run `accpre/eval/online_decode.py` using exactly the loop
pseudocoded in §C.5. Predictor runs in-loop; verifier-on-draft is
skipped entirely; V-free or V-prefix mode per the predictor's family
(§C.3). Emits both throughput and quality on a SINGLE run — no
teacher-forcing, no offline replay.

**L3 — System-level metrics**

| Metric | Definition | Where measured |
|---|---|---|
| `tok_s_online` | `n_new_tokens / elapsed_seconds`, timer wraps the decode loop only (excludes model load, XL-audit, summary writing). | **Online** run. |
| `CF@1` | Per DESIGN.md §D.5.2: fraction of rounds where the predictor's per-round committed output equals strict's per-round committed output under matched randomness. | **Offline replay** on stored Phase-1 strict records (same predictor, same inputs, teacher-forced prefix per round). |
| `xl_audit` | DESIGN.md §D.5.3, **auxiliary**. Side table only. | **Online** run's final generated sequence. |

**Why CF@1 is measured via offline replay and tok/s via the online
run.** The DESIGN.md definition of CF@1 assumes matched randomness and
a shared per-round prefix. Online cheap decode necessarily diverges
from strict after round 0 (different fallback mechanism ⇒ different
round-1 prefix), so a per-round comparison inside the online loop is
ill-posed past round 0. The offline-replay measurement applies the
exact same predictor to each stored strict round with the canonical
prefix of that round — which is what DESIGN.md §D.5.2 prescribes.
Throughput has no such issue; we measure it directly in the online run.

**Semantic clarification — what CF@1 actually measures in v1.** The
current offline-replay reduction (see `accpre/eval/faithfulness.py`)
evaluates `CF@1(round) = 1[L_hat(record) == record.L_strict]`, which is
exact **under the shared-fallback coupling** (both lanes would sample
the same extra token from the same distribution with the same
`fallback_rng` state whenever their L values agree). What this means
in practice:

- CF@1 is a **commit-decision fidelity** metric: "would this
  predictor's commit rule choose the same `L` as strict at strict's
  per-round prefix?"
- It is **not the same as online output-sequence identity.** The
  online cheap-decode lane commits `max(1, L_hat)` tokens with NO
  extra fallback/bonus token (§C.5). Its actual online output
  stream diverges from strict's more aggressively than CF@1
  measures, because the `+1` token strict emits from the verifier
  distribution is simply absent in cheap decode.
- CF@1 is therefore an **upper bound** on actual online output
  identity under the shared-fallback coupling — and it is the
  metric DESIGN intentionally adopts to keep both Pareto axes
  honest and comparable across predictor variants.

Audit issue 5 (2026-04-18 review) confirmed this reduction is
mathematically exact, not approximate. Call-site callers that want
"actual online output identity" should report an additional
trajectory-level token-match statistic; that is deferred (the current
Pareto story uses CF@1 as defined above).

Both measurements are honest and well-defined on their own; the Pareto
figure plots `(tok_s_online, CF@1)` per predictor. This pairs "how fast
in the real cheap loop" with "how faithful the predictor's decisions
would be to strict when evaluated under the canonical shared-prefix
protocol."

**Secondary online diagnostics (reported but not on the Pareto):**

| Metric | Definition |
|---|---|
| `round_commit_mean` | mean tokens committed per round in the online run (≈ `mean(max(1, L̂))`). |
| `generated_length_online` | total new-token length after online decode, per prompt. |
| `seq_token_agreement_strict` | token-wise agreement of the online generated sequence with strict's generated sequence at aligned positions. Decays fast after first round divergence; provided only for a visceral "how soon do outputs diverge" read. |

### H.3 Optional Option-X diagnostic lane

A **diagnostic-only** evaluation where we run the full strict pipeline
(verifier included) and replace only the commit rule with the
predictor. This does NOT skip the verifier, so it is NOT a throughput
lane; its sole purpose is to separate two sources of quality loss:
- "how accurate is the predictor as a commit rule" (what Option X
  measures — its CF@1 equals `L_agreement_strict` under shared
  fallback)
- vs. "how much quality is lost to the missing verifier fallback"
  (the gap between Option-X CF@1 and cheap-online CF@1, if any)

Option X is opt-in per predictor, reported in a side table, and never
on the main Pareto.

### H.4 Every report in one place

Each trained predictor writes exactly one JSON with four top-level
blocks:

- `l1` — offline predictor-level metrics (per §H.1).
- `l2` — offline decision-agreement metrics (per §H.1).
- `l3` — online system-level metrics (per §H.2): `tok_s_online`
  from the online run and `cf_at_1` from offline replay.
- `l3_diag` (optional) — Option-X diagnostic from §H.3 if opted in.

Pareto plotting aggregates `l3`; no other block goes on the Pareto.

---

## I. Engineering constraints (non-negotiable)

1. **All predictor datasets come from Phase 1 logs.** Stage 2A.0
   extends the existing `RoundRecord` with *optional* hidden-state /
   numeric fields (already declared in `accpre/core/schema.py`). No new
   schema version; no new loader; no new serialization format.
2. **No new DDPM code.** Hidden-state extraction lives in
   `accpre/models/drafter_mdlm.py::draft_with_features`, which reuses
   the same `_denoise_step` helper as `draft`. The training-time joint
   drafter path (Stage 2B) must also call that helper. The deferred
   `tests/test_train_infer_consistency.py` (scheduled for Phase 4 in
   DESIGN.md) is brought forward and becomes a **hard gate** for Stage
   2B.1.
3. **No new accept/commit code.** The meta-test already in
   `tests/test_schema.py` continues to enforce that the Leviathan
   pattern appears only in `accpre/core/accept.py`. Predictor training
   MAY import `per_position_accept` for label derivation, never copy
   it.
4. **Common predictor interface.** All predictors inherit from a single
   `PredictorBase` ABC with two methods; each subclass implements at
   least one:
   ```python
   def predict_q2(self, features, **kwargs) -> Tensor   # (gamma,)
   def predict_L(self, features, tau, **kwargs) -> int  # committed length
   ```
   The eval harness consumes either interface uniformly; acceptance
   predictors implement `predict_q2` (and derive `L` via
   `commit_threshold`), length predictors implement `predict_L`.
5. **One canonical feature extractor.** `accpre/collect/features.py`
   is the single source of the feature vectors listed in §C. No
   bespoke extractors in training scripts.
6. **One canonical loss module.** `accpre/train/losses.py` holds MSE
   (Q2), CE (L_τ), and the multitask wrappers. Training scripts pick
   losses by name; no ad-hoc losses in trainers.
7. **One canonical trainer.** `accpre/train/cli.py` is the single
   entry point; it reads a run config (yaml) that names target + input
   family + mode + optional config overrides. No per-predictor
   training scripts. (This is a direct lesson from the old project's
   seven near-duplicate trainers.)

---

## Implementation order (Phase 2)

Exact order. Each item lands and passes its tests before the next
starts.

1. **Stage 2A.0 — collection extension**
   1. `accpre/models/drafter_mdlm.py::draft_with_features`: populate
      the Phase-1 stub by factoring out a shared `_denoise_step` and
      adding a hidden-state forward on the final step (matches the
      logic in old `research/collect_data.py::extract_drafter_hidden`).
   2. `accpre/models/verifier_gpt2.py::prefix_features`: populate the
      Phase-1 stub (verifier last-layer hidden + entropy/margin/top1).
   3. `accpre/collect/features.py`: assembles the four families listed
      in §C from the two module methods above.
   4. `accpre/collect/cli.py`: runs over a split, writes
      `data_collected/stage1.pt` with the extended-but-same-schema
      records. The loader's schema-version assertion is unchanged.
   5. Tests:
      - `tests/test_collect_features.py` — shapes and non-NaN.
      - Extend `tests/test_schema.py` — confirm optional fields
        round-trip.
2. **Stage 2A.1 — frozen acceptance predictors**
   1. `accpre/predictors/base.py`: `PredictorBase` ABC.
   2. `accpre/predictors/acceptance_mlp.py`: feature-MLP predictor
      covering all 3 main-lane families via a single configurable
      head.
   3. `accpre/train/dataset.py`: frozen-features `Dataset`.
   4. `accpre/train/losses.py`: MSE(Q2), BCE(accepted_j), factory.
   5. `accpre/train/cli.py`: one trainer that reads a yaml config.
   6. `accpre/eval/predictor_metrics.py`: L1 metrics.
   7. Hook into `accpre/eval/faithfulness.py` and
      `accpre/eval/decision_agreement.py` (new for Phase 2) for L2/L3.
   8. Run the three `acc_frz_*` configs from §G.
3. **Stage 2A.2 — frozen committed-length predictors (offline)**
   1. `accpre/predictors/length_mlp.py`: τ-conditioned categorical
      head (takes τ as an extra input scalar).
   2. Extend `train/dataset.py` to expose `(features, τ, L_τ)` triples
      (τ sampled per example from the §B.2 distribution).
   3. Extend `train/losses.py` with CE for length.
   4. Extend `eval/predictor_metrics.py` with the length metrics in
      §H.1.
   5. Run the two `len_frz_*` configs from §G (offline L1 + L2).
4. **Stage 2A.3 — online predictor decode (frozen)**
   1. `accpre/eval/online_decode.py`: the canonical online decode
      runner. Implements the loop in §C.5, dispatches on deploy mode
      (V-free vs V-prefix). Reuses `drafter.draft_with_features`,
      `verifier.prefix_features`, `commit_threshold` — no new accept /
      commit code.
   2. `accpre/eval/wallclock.py`: extend to add a `predictor` lane
      variant that calls `online_decode.run_lane(predictor, ...)`
      under the existing Phase 1 timer discipline.
   3. `accpre/eval/faithfulness.py`: usable unchanged — CF@1 is still
      computed offline via replay on strict records with
      `method_commit_fn = predictor_commit(record, τ)`. The online run
      only supplies `tok_s_online` for L3.
   4. `accpre/eval/pareto.py`: add predictor points / curves (one curve
      per predictor when τ is swept).
   5. Run all 5 `*_frz_*` configs in the online-decode lane. Produce
      the Pareto figure including strict + oracle-Q2 (from Phase 1) +
      the 5 frozen predictors.
   6. Tests:
      - `tests/test_online_decode.py` — smoke (V-free and V-prefix
        each run one round on a fixed seed; assert progress-guarantee
        holds when L̂ = 0; assert V-free issues zero verifier calls).
5. **Stage 2A validation gate** — required before any 2B work:
   - All L1 / L2 / L3 reports render; Pareto + auxiliary + L2 tables
     produced; results are sanity-checked by hand (strict CF@1 = 1,
     predictor CF@1 ≤ oracle-Q2 CF@1 at matched τ, etc.).
6. **Stage 2B.1 — joint single-task (offline + online)**
   1. Bring forward `tests/test_train_infer_consistency.py` (the
      Phase-4 anti-drift test) — hard gate before any joint code
      lands.
   2. Extend `drafter_mdlm.py` to expose a grad-enabled forward hook
      on the final DDPM step via the same `_denoise_step` helper.
   3. Add joint MDLM+head predictor classes under
      `accpre/predictors/` (one for Q2, one for L_τ).
   4. Extend `train/cli.py` with the joint mode.
   5. Run `acc_jnt_hid`, `acc_jnt_hidnum`, and `len_jnt_hidnum`.
   6. **Online decode** each joint predictor through the same
      `online_decode.py` from 2A.3 — no new inference code.
7. **Stage 2B.2 — joint multitask (fixed weights) (offline + online)**
   1. Add a `multitask_joint` predictor and a weighted-sum loss.
   2. Run `mt_jnt_hidnum` offline + online.
8. **Stage 2B.3 — learnable weights (offline + online)**
   1. Add learnable `log σ_k` parameters to the multitask loss.
   2. Run `mt_jnt_lw_hidnum` offline + online.
9. **Phase 2 report and freeze.**

Rough wall-clock estimate on Rivanna `gpu-mig 1g.10gb` (small) /
`gpu a100` (joint): 2A ~ 1-2 hrs total, 2B ~ 6-10 hrs total. Stage
2A.0 collection over 80 prompts × 6 actions ≈ one 30-min job.

---

## What should NOT be implemented yet

Explicit non-goals for Phase 2. Each is marked with the earliest stage
it could be revisited.

| Item | Earliest revisit |
|---|---|
| Bernoulli commit rule | future (v2 roadmap) |
| Verifier-free decode lane (Option Y) with predictor-derived fallback | Phase 3 |
| Hybrid / cascade predictor → small-model → XL fallback | Phase 3 |
| Adaptive gamma / T per prefix | Phase 3 |
| Controller / router / retrigger experiments | Phase 3 |
| Per-position ordinal or ranking losses for length | Phase 3 |
| Threshold-aware, margin-aware, focal-style losses | Phase 3 |
| Broad hyperparameter sweeps | Phase 3 |
| Temperature != 1.0 experiments | Phase 3 (requires new protocol + re-collection) |
| q_mode "B" as main experiments (only as single ablation run) | Phase 3 |
| Additional predictor architectures beyond MLP / joint-MLP | Phase 3 |
| Calibration layer (Platt / isotonic) over predictor outputs | Phase 2.5 if `q2_ece` is bad |
| Cross-attention, mini-verifier architectures | out of Phase 2; old project's E4 lane was a negative result |

---

## Summary

- **Stages:** 2A.0 collection → 2A.1/2A.2 frozen offline training →
  **2A.3 online predictor decode** → gate → 2B.1 single-task joint →
  2B.2 multitask fixed → 2B.3 multitask learnable. Each of 2B.1/2B.2/
  2B.3 is offline-trained and then online-decoded in the same loop as
  2A.3.
- **Two-view evaluation:** offline oracle-replay (L1 + L2) and online
  predictor decode (L3). Primary Pareto is `(tok_s_online, CF@1)`;
  `tok_s_online` comes from the online run, `CF@1` from offline replay
  (DESIGN.md §D.5.2 definition).
- **Online decode path (§C.5):** predictor replaces strict's per-
  position verifier comparison. V-free (drafter-hidden-only predictors)
  makes zero verifier calls per round; V-prefix (numeric / hidnum /
  upper-bound predictors) makes exactly one prefix-only verifier
  forward per round. Neither touches the γ-token suffix.
- **Commit convention:** `max(1, L̂)` draft tokens per round; no
  separate bonus / fallback token in cheap decode.
- **Targets:** Q2_j for acceptance, τ-conditioned L_τ^oracle for length.
- **Inputs:** numeric / drafter-hidden / drafter-hidden+numeric as the
  main cheap lane; verifier-prefix-hidden+drafter+numeric as optional
  upper-bound reference.
- **Losses:** MSE for Q2, CE for L_τ; multitask sums in 2B; learnable
  weights only in 2B.3.
- **Hyperparameters:** one default config, no sweeps.
- **Matrix:** 5 frozen + 5 joint/multitask = 10 predictors total, all
  trained offline and run through the same online decode loop.
- **Engineering:** no duplicate draft/verify/accept/commit; no new
  fallback procedure; predictor dataset = Phase-1 `RoundRecord` with
  optional fields populated.
- **Not implemented yet:** Bernoulli commit rule, hybrid fallbacks,
  adaptive γ/T, routers, custom losses, sweeps, temperature ≠ 1.0,
  cross-attention architectures.

Ready for your go-ahead to start Stage 2A.0 (collection extension) when
you're happy with the plan.

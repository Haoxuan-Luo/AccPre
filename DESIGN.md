# AccPre — Phase 0 Design Document (v1)

**Status:** Planning only. No code yet.
**Scope:** v1 only — strict SpecDiff baseline, acceptance predictor variants,
committed-length predictor variants, oracle-Q2 threshold baseline, unified
diagnostics, Pareto plotting. AR speculative decoding is a *reference* lane
only.

---

## Frozen v1 Protocol Decisions

These are locked. All subsequent sections conform to them. No v1 experiment
varies any of these.

1. **No novelty claims.** Acceptance predictor and committed-length predictor
   are treated as **method families** under evaluation, not claimed novel
   contributions. The design does not take a position on their origin.

2. **`q_mode = "A"` is the frozen v1 q definition.** `q_j` is the value of
   the SUBS-parameterized `log p_θ(x₀ | x_t)` row at the DDPM step where
   position j was first unmasked, evaluated on the drafted token, with no
   factor shift. `q_mode = "B"` is retained in the codebase only as a future
   ablation path; it is not exercised in v1 main experiments and does not
   appear in any v1 Pareto figure.

3. **Oracle-Q2 threshold baseline.** The v1 lossy threshold baseline is
   defined as: for each drafted position j, `Q2_j := min(1, p_j / q_j)`
   computed from a full verifier pass; commit the longest prefix with
   `Q2_j ≥ τ` for a given threshold τ. Swept across τ. This is a **strong
   reference baseline**, not a predictor method — it uses the full verifier
   output, so it does not yield a throughput win, but it establishes what a
   threshold-commit rule can achieve with the ideal per-position signal.

4. **Temperature = 1.0 is the main v1 protocol.** `accepted_j` is the
   Bernoulli test `U_j < min(1, p_j/q_j)` at runtime. The `temperature = 0`
   (greedy/argmax) code path exists only for smoke and unit tests, never for
   main experiments or Pareto figures.

5. **Wall-clock throughput only.** The Pareto y-axis is
   `wallclock_tok_s`, measured with the decode-loop timer described in §D.5.
   All offline-timing formulas (including the old project's
   `t_real = draft + verify·P/(P+G) + ε`) are **retired** from the main
   evaluation. They may appear in internal diagnostics with a written
   caveat; they are never on a reportable plot.

6. **Controlled Faithfulness@1 (CF@1) is the primary Pareto quality axis.**
   CF@1 measures, under matched randomness and a shared draft proposal,
   the fraction of rounds where the method produces the *same committed
   output sequence* as strict SpecDiff. Formally defined in §D.5. XL-audit
   quality is retained only as an **auxiliary** diagnostic, reported in a
   separate table, never on the Pareto axis.

---

## Residual open questions (not blocking Phase 1)

1. **CF@1 operational granularity.** v1 defines CF@1 as round-level output
   identity under matched randomness (see §D.5). A finer per-token variant
   (`CF@k`, agreement on the first k committed tokens of a round) is
   conceivable. Flagging for future consideration; Phase 1 uses round-level
   only.
2. **AR reference lane on CF@1 axis.** CF@1 is defined relative to strict
   SpecDiff's draft proposal. AR speculative decoding uses a different
   drafter and does not share a draft with strict SpecDiff, so CF@1 is not
   meaningfully defined for the AR lane. v1 plots AR on a **separate panel**
   with only wall-clock + auxiliary XL-audit; CF@1 is a SpecDiff-family
   metric.
3. **Strict vs lossy fallback symmetry.** v1 decides that the oracle-Q2
   threshold baseline uses the exact same fallback/bonus sampling path as
   strict SpecDiff (same adjusted distribution, same shared RNG state).
   Under this coupling, CF@1 reduces to `1[L_method == L_strict]` per round.
   This is not an approximation; it is what the matched-randomness coupling
   guarantees. Documented here to pre-empt confusion.

---

## A. Proposed directory structure

```
AccPre/
├── README.md                         # short project overview + how to run v1
├── DESIGN.md                         # this file
├── pyproject.toml                    # package metadata + pinned deps
├── configs/
│   ├── protocol.yaml                 # THE canonical protocol (see §D)
│   ├── collect.yaml                  # sweep config for Stage 1
│   └── eval.yaml                     # sweep config for Stage 3
├── accpre/                           # source package
│   ├── __init__.py
│   ├── models/                       # thin wrappers over HF models
│   │   ├── drafter_mdlm.py           # MDLMDrafter
│   │   └── verifier_gpt2.py          # GPT2Verifier
│   ├── core/                         # single source of truth for decode mechanics
│   │   ├── protocol.py               # ProtocolConfig dataclass
│   │   ├── schema.py                 # RoundRecord + save/load + provenance
│   │   ├── draft_verify.py           # one canonical draft→verify round
│   │   ├── accept.py                 # one canonical per-position accept test
│   │   └── commit.py                 # commit_strict, commit_threshold
│   ├── data/
│   │   ├── prompts.py                # load_owt_prompts (reused)
│   │   └── splits.py                 # canonical train/val/test split
│   ├── collect/
│   │   ├── features.py               # verifier + drafter feature extractors
│   │   └── cli.py                    # Stage-1 entry point
│   ├── predictors/
│   │   ├── base.py                   # PredictorBase ABC
│   │   ├── acceptance_mlp.py         # scalar-feature MLP predictor
│   │   ├── acceptance_joint_probe.py # MDLM + head probe (shares backbone)
│   │   ├── mstar_survival.py         # monotone survival curve head
│   │   └── mstar_direct.py           # direct categorical m* head
│   ├── train/
│   │   ├── dataset.py                # RoundRecord → torch Dataset
│   │   ├── losses.py                 # masked BCE, mstar CE, monotone survival
│   │   └── cli.py                    # Stage-2 entry point
│   ├── eval/
│   │   ├── predictor_metrics.py      # L1: AUROC / Brier / ECE
│   │   ├── decision_agreement.py     # L2: L-agreement vs strict + vs oracle-Q2
│   │   ├── wallclock.py              # L3: end-to-end tok/s (Pareto y-axis)
│   │   ├── faithfulness.py           # L3: CF@1 (Pareto x-axis, primary)
│   │   ├── quality.py                # ONE xl_audit (AUXILIARY only, not Pareto)
│   │   └── pareto.py                 # frontier + plotting
│   └── reference/
│       └── ar_speculative.py         # romsto-style AR baseline (L3 only)
├── scripts/                          # thin shell shims over CLIs
│   ├── collect_stage1.sh
│   ├── train_predictor.sh
│   └── run_eval.sh
├── checkpoints/                      # .gitignored; predictor weights + preds
├── data_collected/                   # .gitignored; RoundRecord .pt files
├── figures/                          # .gitignored; output plots
└── tests/
    ├── test_accept.py                 # Phase 1
    ├── test_commit.py                 # Phase 1
    ├── test_schema.py                 # Phase 1
    ├── test_smoke_pipeline.py         # Phase 1 (strict + oracle-Q2 smoke)
    ├── test_faithfulness.py           # Phase 1 (CF@1)
    ├── test_matched_randomness.py     # Phase 1 (RNG replay)
    └── test_train_infer_consistency.py  # Phase 4 (deferred)
```

Design rules enforced by this structure:

- `accpre/core/` is the **only** place that implements draft→verify→accept→
  commit. Everyone else (baselines, predictors, rules, eval) imports from
  here. No duplicate accept/commit logic is allowed; tests (see §I) enforce
  this by construction.
- The three eval layers live in three different files (`predictor_metrics`,
  `decision_agreement`, `wallclock`) to make the separation structural, not
  just stylistic.
- `reference/ar_speculative.py` is isolated so AR does not leak into the main
  storyline. It only gets called by `eval/wallclock.py` as a lane, never by
  anything else.
- No `research/` dir, no `scripts/` containing logic — all logic lives in
  `accpre/`, scripts are thin shims.

---

## B. Module responsibilities

**`accpre/models/drafter_mdlm.py`**
Wraps the pretrained MDLM. Exposes:
- `draft(prefix, gamma, T, protocol) → (draft_tokens, q_log_probs)` — no_grad,
  eval-time. Returns one `q_log_probs[j]` row per drafted position, computed
  per the pinned `protocol.q_mode` at the commitment step.
- `draft_with_features(prefix, gamma, T, protocol) → (draft_tokens, q_log_probs,
  drafter_hidden, draft_time)` — used by Stage-1 collection. Runs **exactly
  the same denoising schedule**; the hidden state comes from a separate final-
  step forward pass so logits used for `q` are not recomputed.
- Internal helper `_denoise_step(logits, xt, t_now, t_next, q_mode)` is a pure
  function, callable with or without grad. This is the anti-drift fix for
  audit risk #7: the joint probe predictor (training-time) will call this
  same helper, so training-time drafting cannot diverge from inference-time
  drafting.

**`accpre/models/verifier_gpt2.py`**
Thin wrapper. `score(token_ids) → log_probs` (shape `(seq_len, vocab)`) and
`prefix_features(prefix_ids) → (hidden, numeric_dict)` for Stage-1. Identical
semantics to the old `GPT2XLVerifier`.

**`accpre/core/protocol.py`**
```
@dataclass(frozen=True)
class ProtocolConfig:
    temperature: float = 1.0
    q_mode: str = "A"                 # "A" or "B"
    schema_version: int = 1           # bump if RoundRecord layout changes
    drafter_model: str = "kuleshov-group/mdlm-owt"
    verifier_model: str = "gpt2-xl"
    dtype: str = "float32"
    max_verifier_ctx: int = 1024
```
All Stage-1, Stage-2, Stage-3 code receives a `ProtocolConfig` and is required
to stamp it into every artifact. Mismatches are asserted at load time.

**`accpre/core/schema.py`**
`RoundRecord` dataclass — every Stage-1 round is one `RoundRecord`. Fields
in §C step 7. `save_records(list[RoundRecord], path)` and
`load_records(path) → list[RoundRecord]` are the only supported I/O paths.
`load_records` asserts `schema_version` and `protocol` consistency.

**`accpre/core/accept.py`**
Exactly one function:
```
def per_position_accept(draft_tokens, q_log_probs, target_log_probs,
                        prefix_len, protocol, rng) -> AcceptOutcome
```
where `AcceptOutcome` has `accepted_j`, `survived_j`, `q_j`, `p_j`,
`min_pq_j`. This is the **only** implementation of the Leviathan accept test.
It branches on `protocol.temperature` (0 → argmax, >0 → Bernoulli). Every
caller — strict baseline, Stage-1 collection, lossy evaluator — imports this
function. No other file may implement it.

**`accpre/core/commit.py`**
```
def commit_strict(accepted_j) -> int           # first 0, else gamma
def commit_threshold(alpha, tau) -> int        # first j with alpha[j] < tau
```
Both return `L ∈ [0, gamma]`. That's it. No other commit rules in v1.
(Bernoulli rule stub is a one-line `raise NotImplementedError` with a comment
pointing to the v2 plan.)

**`accpre/core/draft_verify.py`**
```
def draft_verify_round(prefix_ids, drafter, verifier, gamma, T, protocol,
                       feature_extractors=None, rng=None) -> RoundRecord
```
Composes: `drafter.draft_with_features(...)` → `verifier.score(...)` →
`per_position_accept(...)` → `commit_strict(...)`. Returns a fully populated
`RoundRecord`. Used by both Stage-1 collection and Stage-3 strict baseline
wall-clock runs. `feature_extractors=None` skips hidden-state extraction for
speed.

**`accpre/data/prompts.py`**
Reused verbatim from old project.

**`accpre/data/splits.py`**
One canonical train / val / test split of prompts, fixed seed, documented.
Addresses audit risk #4 (multiple splits in play).

**`accpre/collect/*`**
Stage 1. Loops (prompt, action) × n_rounds, calls
`draft_verify_round(..., feature_extractors=stage1)`, writes RoundRecords.

**`accpre/predictors/*`**
Each predictor is a `PredictorBase` subclass with:
- `predict_alpha(record_or_batch) → ᾱ_j` (acceptance predictors)
- `predict_mstar(record_or_batch) → m_hat ∈ [0, gamma]` (committed-length
  predictors)
An acceptance predictor can derive `m_hat` via a rule
(`commit_threshold(ᾱ, tau)`); a committed-length predictor exposes `m_hat`
directly. Some predictors expose both.

**`accpre/train/*`**
Stage 2. Dataset wraps RoundRecords; losses masked by `survived_j`; CLI
trains one predictor at a time. Training writes: `checkpoints/{name}/model.pt`,
`checkpoints/{name}/preds_val.pt`, `checkpoints/{name}/config.yaml`
(echoes the protocol + predictor hyperparams).

**`accpre/eval/predictor_metrics.py` (Level 1)**
Loads saved per-position predictions + RoundRecords from the **val/test
split**, reports AUROC / Brier / ECE on labels `accepted_j` masked by
`survived_j`. For committed-length predictors: `mstar_mae` and
`mstar_exact_match`. Does **not** depend on any commit rule.

**`accpre/eval/decision_agreement.py` (Level 2)**
For each round, computes `L_hat` (from predictor + commit rule) and compares
against `L_strict` (from the record) and `L_tau` (from lossy threshold rule on
oracle `min(1,p/q)`). Reports exact-match, overcommit-count, undercommit-count.
**This is the primary evaluation lane for the research question**, per the
user's design priority.

**`accpre/eval/wallclock.py` (Level 3, throughput axis)**
Runs each method end-to-end on held-out prompts. Times the decode loop only
(excludes model load, excludes post-hoc audit, excludes CF@1 computation).
Emits per-round `RoundRecord`s alongside the timing — those records are the
input to `faithfulness.py`. Methods available as lanes: `strict_specdiff`,
`oracle_q2_threshold(tau)` (offline-comparable, not a throughput win —
included for reference), `predictor_<name>` (Phase 3+), `ar_reference`
(separate panel, Phase 6).

**`accpre/eval/faithfulness.py` (Level 3, primary quality axis)**
Computes Controlled Faithfulness@1 (CF@1) per §D.5. Takes
`records_strict: list[RoundRecord]` and a `method_commit_fn: RoundRecord → int`
(the method's committed length given the strict record's logged inputs).
Returns per-prompt CF@1 scalars and an aggregate with per-prompt bootstrap CI.
Pure offline function of stored records — no model calls. Under the shared-
fallback coupling declared in the residual questions, CF@1 reduces to
`mean_round(1[method_L == strict_L])`, but the function is written in its
general form so future methods with divergent fallbacks can be evaluated
without API change.

**`accpre/eval/quality.py` (auxiliary only)**
Exactly one `xl_audit(generated_ids, prefix_len, verifier) -> float`.
Addresses audit risk #2 (five copies, two incompatible signatures). This one
uses the `prefix_len` signature (audits all post-prefix tokens). **Not on the
Pareto axis.** Reported only in a side table for lane sanity-checks.

**`accpre/eval/pareto.py`**
Aggregates `wallclock.py` + `faithfulness.py` outputs, computes per-prompt
bootstrap CIs, draws the Pareto frontier. Inputs: list of
`(method, wallclock_tok_s, cf_at_1)` per prompt. XL-audit quality is attached
to the same rows but plotted only as a separate auxiliary panel. No decoding
logic here — pure plotting and stats.

**`accpre/reference/ar_speculative.py`**
Minimal romsto-aligned AR speculative-decoding runner, exposed only as a lane
in `eval/wallclock.py`.

---

## C. End-to-end data flow

```
[config]
  ProtocolConfig  (temperature=1.0, q_mode="A", schema_version=1, ...)

[Stage 1 — offline data collection]

  load_owt_prompts(n, prefix_len, seed)
    → list[(prefix_ids, prefix_text)]

  for each prompt:
    verifier_prefix_features(verifier, prefix_ids)
      → (verifier_hidden, {entropy, margin, top1_prob})

    for each action (gamma, T):
      for each round:
        drafter.draft_with_features(prefix_ids, gamma, T, protocol)
          → (draft_tokens, q_log_probs, drafter_hidden, draft_time)

        candidate = concat(prefix_ids, draft_tokens)
        verifier.score(candidate)
          → target_log_probs (seq_len, vocab)

        round_rng_seed = derive_seed(protocol, prompt_idx, round_idx)
        rng = Generator(round_rng_seed)

        per_position_accept(draft_tokens, q_log_probs, target_log_probs,
                            prefix_len, protocol, rng)
          → accepted_j, survived_j, q_j, p_j, min_pq_j, U_j

        L = commit_strict(accepted_j)

        bonus_or_fallback_token = sample_fallback_or_bonus(
          L, gamma, draft_log_probs, target_log_probs, prefix_len, rng,
        )

        record = RoundRecord(
          prompt_idx, round_idx, gamma, T, prefix_len,
          draft_tokens, q_j, p_j, min_pq_j, accepted_j, survived_j, U_j, L,
          bonus_or_fallback_token, round_rng_seed,
          draft_time, verify_time,
          verifier_hidden, drafter_hidden, prefix_tail_32,
          verifier_entropy, verifier_margin, verifier_top1_prob,
          protocol=protocol,
        )

  save_records(all_records, "data_collected/stage1.pt")

[Stage 2 — predictor training]

  load_records("data_collected/stage1.pt")
    → assert schema_version == 1, protocol == expected
  splits.train_val_test(records)
    → (train, val, test)

  Dataset yields either:
    - per-position samples (one per (round, j) with survived_j == 1) for
      acceptance predictors, OR
    - per-round samples for committed-length predictors.

  train loop:
    for batch in train:
      pred = predictor(batch.features)          # e.g. ᾱ_j or p(m*=k)
      loss = masked_bce_or_mstar_ce(pred, batch.labels, mask=batch.survived)
      optimizer.step()

  after training:
    - save checkpoints/{name}/model.pt
    - compute preds on val + test, save preds_val.pt, preds_test.pt
    - echo protocol into config.yaml

[Stage 3 — evaluation]

  L1 predictor metrics (offline, cheap):
    load preds_test.pt + records_test
    → AUROC / Brier / ECE on accepted_j|survived_j
      (and mstar_mae / mstar_exact_match if applicable)

  L2 decision agreement (offline):
    for each test round:
      L_strict   = record.L                                      # from record
      L_oracleQ2 = commit_threshold(record.min_pq_j, tau)         # oracle baseline
      L_hat      = commit_threshold(ᾱ_j_predicted, tau)  or  predictor.predict_mstar(...)
    → L-agreement, overcommit, undercommit, per-method tables

  L3 wall-clock + CF@1 (the headline Pareto):
    throughput (y-axis):
      for each held-out prompt:
        for each method in {strict, oracle_q2_threshold, predictor_*}:
          run the method's decode loop, timing only the loop
          tok_s = n_new / elapsed
    CF@1 (x-axis, primary):
      records_strict = strict records on the same held-out prompts
      for each method:
        method_commit_fn = the method's per-round L decision given logged inputs
        cf_at_1 = faithfulness.controlled_faithfulness_at_1(
                    records_strict, method_commit_fn)
    auxiliary:
      quality_xl = xl_audit(final_ids, prefix_len, verifier)    # side table
    → pareto.py builds (tok_s, CF@1) frontier; XL-audit on auxiliary panel
      AR reference lane: separate panel, (tok_s, XL-audit) only
```

Every arrow in this diagram corresponds to exactly one function in exactly one
module. No step is implemented twice.

---

## D. Exact definitions we will use

These supersede any definition in the old project.

### D.1 Per-position quantities (computed by `core/accept.py`)

| Symbol | Definition |
|---|---|
| `q_j` | `exp(q_log_probs[j, draft_tokens[j]])`. `q_log_probs[j]` is the SUBS-parameterized row at the DDPM step where position j was first unmasked, produced by the drafter under `protocol.q_mode`. Canonical default: `q_mode = "A"` (no log-factor shift). |
| `p_j` | `exp(target_log_probs[prefix_len - 1 + j, draft_tokens[j]])`. The verifier's next-token probability at the position *preceding* draft position j. |
| `min(1, p_j/q_j)` | `min(1.0, p_j / q_j)` with both clamped at `1e-10`. Not its own random variable; just the Leviathan acceptance probability. |
| `accepted_j` | Binary. Under `temperature > 0`: `1 if U_j < min(1, p_j/q_j) else 0`, `U_j ~ Uniform(0,1)` drawn fresh per position. Under `temperature == 0`: `1 if argmax(target_log_probs[prefix_len-1+j]) == draft_tokens[j] else 0`. The protocol pins *which* rule; the definition is not temperature-polymorphic at runtime. |
| `survived_j` | Binary. `survived_0 = 1`; `survived_j = accepted_{j-1} AND survived_{j-1}` for j > 0. |

### D.2 Per-round quantities

| Symbol | Definition |
|---|---|
| `L` (a.k.a. `m*`, committed length) | `commit_strict(accepted_j) = first j with accepted_j == 0, else gamma`. Range `[0, gamma]`. Equivalent algebra: `L = sum_j survived_j * accepted_j`. |
| `n_committed_with_bonus` | `L + 1` — one bonus (full-accept) or fallback (partial-accept) token per round. Used only in throughput; excluded from acceptance-rate aggregates. |
| `acceptance_rate` (per-round) | `L / gamma`. **NOT** `sum(accepted_j)/gamma` (those can differ when a late position happens to accept after an earlier reject — the Leviathan rule prohibits committing it, so only contiguous prefix counts). |
| `U_j` | Per-position uniform draws used by the Bernoulli accept test under the main v1 protocol (`temperature = 1.0`). **Logged in every record** so CF@1 can replay decisions under matched randomness. |
| `round_rng_seed` | Integer seed used to derive `U_j` and the fallback/bonus RNG state for a given round. Logged in every record. Derived deterministically from `(protocol, prompt_idx, round_idx)` so replays are reproducible. |

### D.3 Predictor-level accuracy metrics — LEVEL 1

Computed by `eval/predictor_metrics.py`. Pure function of `(predictions,
labels)`. No commit rule involved.

| Metric | Definition |
|---|---|
| `accept_auc` | AUROC of `ᾱ_j` against `accepted_j` over all per-position samples with `survived_j == 1`. Masking is **required** — positions with `survived_j == 0` do not have a defined conditional-acceptance label. |
| `accept_brier` | Mean of `(ᾱ_j - accepted_j)^2` over survived positions. |
| `accept_ece` | Expected calibration error, 10 equal-mass bins, over survived positions. |
| `mstar_mae` | Mean of `|m_hat - L|` over all rounds. Applicable to committed-length predictors. |
| `mstar_exact_match` | Fraction of rounds with `m_hat == L`. |

### D.4 Decision-level agreement metrics — LEVEL 2

Computed by `eval/decision_agreement.py`. A function of
`(predictor_output, commit_rule, baseline_output)` per round. Pure offline
function of stored records.

Let `L_strict` = ground-truth committed length from the record.
Let `L_hat` = committed length induced by the method under its commit rule.

| Metric | Definition |
|---|---|
| `L_agreement_strict` | Fraction of rounds with `L_hat == L_strict`. Note: under v1's shared-fallback coupling this equals CF@1 numerically; it is still reported separately because it lives in the L2 decision-level panel, not on the Pareto. |
| `L_overcommit_rate_strict` | Fraction with `L_hat > L_strict` (would commit a token the strict rule rejects). |
| `L_undercommit_rate_strict` | Fraction with `L_hat < L_strict` (wastes accepted tokens). |
| `L_overcommit_token_count` | Total tokens committed beyond `L_strict`, summed over all rounds. The "quality-violating" budget. |
| `L_agreement_oracleQ2(τ)` | Same metric family but with `L_τ = commit_threshold(min_pq_j, τ)` on the oracle Q2 signal as the reference point. Used to compare predictor agreement with the oracle-Q2 threshold baseline. |

Per the user's design priority: **L2 is the primary research lane** for the
predictor comparisons. L1 is auxiliary calibration. L3 is the headline Pareto
(throughput + CF@1).

### D.5 System-level quality/throughput metrics — LEVEL 3

Computed by `eval/wallclock.py`, `eval/faithfulness.py`, and
`eval/quality.py`.

#### D.5.1 Throughput (Pareto y-axis)

| Metric | Definition |
|---|---|
| `wallclock_tok_s` | `n_new / elapsed_seconds`. Timer wraps the decode loop only; excludes model load, excludes post-hoc quality audits, excludes CF@1 computation. Same timer specification for every lane. |

#### D.5.2 Controlled Faithfulness@1 (Pareto x-axis, primary)

Let `S_strict(round)` denote the ordered sequence of tokens strict SpecDiff
would append to the running context in round r, under a fixed protocol and
fixed per-round randomness: specifically
`S_strict(round) = draft[:L_strict] ++ [bonus_or_fallback_strict]`, where
`L_strict = commit_strict(accepted_j^strict)` and `accepted_j^strict` is the
Bernoulli outcome of `U_j < min(1, p_j/q_j)` for `U_j ~ Uniform(0,1)` drawn
from the round's RNG stream. The `bonus_or_fallback_strict` is sampled from
the shared-fallback protocol: target-tail on full-accept, `(p − q)_+`
adjusted-distribution on partial-accept, using a stored fallback RNG state.

Let `S_method(round)` denote the analogous ordered output sequence produced
by the method under test, **using the exact same inputs** (`prefix`,
`draft_tokens`, `q_log_probs`, `target_log_probs`) and the **exact same**
per-position `U_j` and fallback RNG state.

| Metric | Definition |
|---|---|
| `CF@1(round)` | `1` if `S_method(round)` is element-wise identical to `S_strict(round)`, else `0`. |
| `CF@1` (aggregate) | Mean of `CF@1(round)` over all rounds in the eval set, with per-prompt bootstrap CI. |

**Matched-randomness discipline (normative).** For CF@1 to be well-defined,
strict and every compared method must share a specific set of inputs and
random draws. v1 enforces the following discipline by construction in
`core/draft_verify.py`.

*What is shared (identical inputs to both strict and the compared method):*

| Shared quantity | Source | How it is guaranteed shared |
|---|---|---|
| `prefix_ids` (the running context at round start) | Caller | Both lanes pass the same prefix into the round. |
| `draft_tokens`, `draft_log_probs` (q_j values) | Drafter | Both lanes use the same MDLM drafter with the same `draft_rng` (see below) → identical sampling trajectory → identical draft proposal. |
| `target_log_probs` (p_j values) | Verifier | Verifier is deterministic given `(prefix, draft_tokens)`. |
| `U_j` (the per-position Bernoulli draws) | `accept_rng` | Same generator seed + same number of draws → identical `U_j` stream. Strict uses it to decide acceptance; lossy/oracle-Q2 draws and *logs* it but doesn't use it for the commit decision. |
| Fallback RNG state at the moment fallback is sampled | `fallback_rng` | `fallback_rng` is a **separate generator**, never consumed by accept. Its state is therefore identical for both lanes regardless of what the commit rule decided. |

*What is replayed / derived from `round_rng_seed`:*

```
round_rng_seed                                (logged in every RoundRecord)
    │
    ├── draft_rng    = Generator(seed XOR protocol.draft_salt)
    │       └── feeds drafter.draft(..., generator=draft_rng)
    │
    ├── accept_rng   = Generator(seed XOR protocol.accept_salt)
    │       └── feeds per-position U_j draws in per_position_accept
    │
    └── fallback_rng = Generator(seed XOR protocol.fallback_salt)
            └── feeds bonus-or-fallback token sampling
```

`round_rng_seed` itself is derived deterministically from
`SHA-256(protocol.fingerprint() | prompt_idx | round_idx)` — stable across
Python processes, stable across machine restarts, unique per round under a
fixed protocol.

*Draft-proposal guarantee:* under v1, every SpecDiff-family lane uses the
same drafter model (`protocol.drafter_model`), the same `(gamma, T)`, and
the same `draft_rng`. `drafter.draft` accepts `generator=draft_rng` and
routes it into every `torch.multinomial` call inside the DDPM loop (both
intermediate steps and the final-step sampling). Two lanes with the same
`round_rng_seed` therefore produce **identical** `draft_tokens` and
`draft_log_probs`. The oracle-Q2 threshold baseline is not an exception —
it uses the same drafter call.

*Fallback / bonus path under matched randomness:*

- Strict SpecDiff: commits `draft_tokens[:L_strict]`, then samples one
  extra token from `fallback_rng`. The distribution is target-tail on
  full-accept (`L == gamma`) or `norm((p − q)_+)` on partial-accept.
- Oracle-Q2 threshold (and any other v1 method): identical extra-token
  sampling procedure, driven by the **same** `fallback_rng` generator
  state, evaluated at position `L_method`.

Consequence: if `L_method == L_strict` then the fallback position is the
same, the distribution there is computed from the same shared inputs
(`target_log_probs`, `draft_log_probs`, `prefix_len`), and the generator
is in the same state → the sampled extra token is identical → the full
round output is identical → `CF@1(round) = 1`. If `L_method ≠ L_strict`,
either the committed draft-prefix length differs or the extra-token
position differs, so the output sequences differ and `CF@1(round) = 0`.

**Reduction:** Under the v1 shared-fallback coupling,
`CF@1(round) ≡ 1[L_method == L_strict]`. This makes CF@1 computable
offline from stored strict records — no re-decoding required. The
`eval/faithfulness.py` API still accepts an arbitrary `method_commit_fn`
and an optional fallback-comparison hook, so a future method that breaks
the shared-fallback coupling (e.g., uses a different adjusted
distribution) remains evaluable without API change.

**What is NOT shared** (this is the point of the comparison):

- The commit rule itself. Strict uses `commit_strict(accepted_j)`;
  oracle-Q2 uses `commit_threshold(min_pq_j, τ)`; predictors use their
  own rule. Different rules may yield different `L` values — that is
  precisely what CF@1 measures.

**CF@1 as a distributional measure.** For a method whose commit decision is
deterministic given `(prefix, draft, q, p)`, `E[CF@1]` over the RNG is
exactly the probability that the method's single-round output draws from the
same point in sample space as strict. Aggregated over rounds and prompts, it
is a Monte Carlo estimate of the total output-distribution mass the method
preserves relative to strict.

**Scope of CF@1 in v1:** defined for all SpecDiff-family lanes. *Not* defined
for the AR reference lane (different drafter, no shared draft proposal); AR
is plotted on a separate panel with wall-clock + auxiliary XL-audit only.

#### D.5.3 Auxiliary quality metrics (not on the Pareto)

| Metric | Definition | Role |
|---|---|---|
| `xl_audit_quality` | `(1 / (len(gen) − prefix_len)) * sum_k 1[gen[k] == argmax(verifier.score(gen[:k])[−1])]` for `k ∈ [prefix_len, len(gen))`. Post-hoc, not causal to decode. One canonical implementation. | **Auxiliary only.** Side-table sanity check on each lane. Never on Pareto axes. |

The three levels (L1, L2, L3) are reported as three separate tables / three
separate plot panels. They are never averaged together and never conflated.

---

## E. Old files to reuse directly

Files the audit flags as trusted and which align with the new design (copy
into `accpre/` with minimal adaptation — mostly package path rewrites):

| Old path | New path | Notes |
|---|---|---|
| `drafter/mdlm_drafter.py` | `accpre/models/drafter_mdlm.py` | Keep the `q_mode` branching and the SUBS logic. Add `draft_with_features(...)` method by merging the logic from old `collect_data.py::extract_drafter_hidden` into this module, so draft logic lives in one file. Factor `_denoise_step` into a shared helper. |
| `verifier/gpt2xl_verifier.py` | `accpre/models/verifier_gpt2.py` | Add a `prefix_features(prefix_ids)` method that returns the last-layer hidden state + `(entropy, margin, top1_prob)` tuple — merging old `collect_data.py::extract_verifier_features` into this module. |
| `data/prompts.py` | `accpre/data/prompts.py` | Verbatim. |
| `research/lossy_decode.py` (just `threshold_acceptance`) | `accpre/core/commit.py` (as `commit_threshold`) | Copy only the function body. Drop `bernoulli_acceptance` (v2) and `threshold_sweep` (folded into `eval/decision_agreement.py`). |
| `third_party/romsto/` | `accpre/reference/third_party/romsto/` | Vendored. Only used by `reference/ar_speculative.py`. |

---

## F. Old files that are references only (ideas, not safe to copy)

| Old path | Why reference-only | What to mine it for |
|---|---|---|
| `pipeline/specdiff_pipeline.py` | Strict-SpecDiff implementation is correct but bundles accept logic, commit logic, EOS handling, bonus/fallback sampling, per-position stats, and warm-start stubs into one class. Audit risk #5 (no EOS stop) also lives here. | Extract the decode *outer loop* and `_sample_adjusted`. Rebuild as `eval/wallclock.py::decode_strict` that calls `core/draft_verify.py` in a loop. Fix the EOS handling so strict and AR lanes terminate symmetrically. |
| `research/collect_data.py` | Duplicates drafter DDPM logic inside `extract_drafter_hidden`. Target alignment with the canonical drafter is not guaranteed. | Mine it for the RoundRecord fields, provenance stamp, and the two-pass hidden-state trick. Do not copy the DDPM reimplementation — replace with `drafter.draft_with_features(...)`. |
| `research/joint_train.py` | 1304 lines. `JointModel` re-implements drafting inline with different commit semantics (audit risk #7). | Mine it for the `JointAcceptanceHead` architecture and probe/joint schedule. Rebuild the training loop around `predictors/acceptance_joint_probe.py` that calls the shared `_denoise_step` helper. |
| `research/lossy_pareto_util.py` | Contains the offline timing formula that systematically over-credits methods skipping verifier (audit risk #3). | Mine it for the bootstrap CI and the per-prompt aggregation. Drop the synthetic `t_real` formula; v1 uses wall-clock only. |
| `research/unified_predictor_eval.py` | Audit calls it the best eval template. | Mine for the per-round alignment logic and output JSON schema. Rebuild split into L1 / L2 files per §D. |
| `research/models/accept_predictor.py`, `mstar_head.py`, `survival_head.py`, `tiny_qualifier.py` | Model architectures are fine in isolation. | Copy the `nn.Module` class bodies into `predictors/*.py`; drop whatever training wrappers live around them. |
| `scripts/run_unified_wallclock.py`, `run_predictor_screen.py` | Correct wall-clock but contain duplicated `xl_audit` and duplicated `summarize`. | Mine for the per-method lane structure. The loop itself becomes `eval/wallclock.py`; the five `xl_audit` duplicates collapse into `eval/quality.py`. |

---

## G. Parts to rebuild from scratch

Anything not listed in E or F is rebuilt. Specifically:

1. **`core/protocol.py`** — new dataclass; old project had protocol info
   scattered across argparse defaults.
2. **`core/schema.py`** — new `RoundRecord` dataclass + assertive save/load.
   Old project used a dict of dicts with per-record provenance patched on.
3. **`core/accept.py`** — new single function, merging the three near-copies
   in `specdiff_pipeline.py::_accept_reject`, `collect_data.py::run_one_round`,
   and `lossy_pareto_util.py` inline loops.
4. **`core/draft_verify.py`** — new. Composes `drafter.draft_with_features` +
   `verifier.score` + `per_position_accept` + `commit_strict`. Nothing
   analogous exists as a single function in the old project.
5. **`collect/cli.py`** — new CLI structured around `ProtocolConfig`.
6. **`predictors/base.py`** — new ABC. Old project had five predictor
   architectures with five different interfaces and training scripts.
7. **`train/*.py`** — new training harness centered on `RoundRecord` +
   `PredictorBase` + one masked-BCE / one mstar-CE / one monotone-survival
   loss in `losses.py`.
8. **`eval/predictor_metrics.py`**, **`eval/decision_agreement.py`**,
   **`eval/wallclock.py`**, **`eval/quality.py`**, **`eval/pareto.py`** —
   all new, each with a single responsibility per §B.
9. **`reference/ar_speculative.py`** — new thin wrapper around romsto. Old
   `pipeline/aligned_baseline.py` is close but has its own timing
   conventions; rebuild to match the v1 timing protocol.
10. All tests in `tests/` — new.

---

## H. Minimal implementation order (phased)

**Phase 1 — Core + strict baseline + oracle-Q2 baseline + CF@1.**
Target: strict SpecDiff and oracle-Q2 threshold both run end-to-end; CF@1 and
wall-clock both exist; XL-audit exists as auxiliary only. No predictor code.
Detailed Phase 1 file list and test list are in the Phase 1 implementation
plan appended at the end of this document.
  — **Exit criteria (all must hold):**
    1. Strict SpecDiff runs end-to-end and emits `RoundRecord`s with full
       logging schema including `U_j`, `round_rng_seed`, and
       `bonus_or_fallback_token`.
    2. Oracle-Q2 threshold baseline runs end-to-end (evaluated offline on
       strict records) for a τ sweep.
    3. `eval/wallclock.py` measures the decode loop only for the strict lane.
    4. `eval/faithfulness.py::controlled_faithfulness_at_1` computes CF@1
       from stored records and returns per-prompt + aggregate values.
    5. `eval/quality.py::xl_audit` exists with a single canonical signature
       and is reported in a clearly-labeled auxiliary side table, never on
       the Pareto axes.
    6. All Phase 1 tests pass (see §I).

**Phase 2 — Stage-1 collection for predictors.**
  2.1. `collect/features.py`, `collect/cli.py` → writes `stage1.pt`. Adds
  hidden-state fields on top of the Phase-1 `RoundRecord` schema.
  2.2. `eval/decision_agreement.py` with `L_oracleQ2(τ)` lane populated.
  — **Exit criterion:** `stage1.pt` loads under the same schema loader,
  `decision_agreement` reports `L_agreement_strict` vs `L_oracleQ2` on val.

**Phase 3 — First acceptance predictor.**
  3.1. `predictors/base.py`, `predictors/acceptance_mlp.py` (scalar + hidden
  features).
  3.2. `train/dataset.py`, `train/losses.py` (masked BCE only),
  `train/cli.py`.
  3.3. `eval/predictor_metrics.py` (L1 metrics).
  — **Exit criterion:** trained MLP predictor with sensible AUROC; L2
  agreement plot exists.

**Phase 4 — Joint-probe acceptance predictor.**
  4.1. `predictors/acceptance_joint_probe.py`, exercising the shared
  `_denoise_step` helper.
  4.2. `tests/test_train_infer_consistency.py` — asserts the joint-probe's
  draft path produces **identical** tokens and q to `drafter.draft(...)` at
  eval time (the anti-divergence test for audit risk #7).
  — **Exit criterion:** consistency test passes on a fixed seed.

**Phase 5 — Committed-length predictor variants.**
  5.1. `predictors/mstar_survival.py` + monotone survival loss.
  5.2. `predictors/mstar_direct.py` + mstar-CE loss.
  5.3. `eval/predictor_metrics.py` gains `mstar_mae`, `mstar_exact_match`.
  — **Exit criterion:** two committed-length predictors trained and
  agreement-evaluated.

**Phase 6 — AR reference lane + full Pareto.**
  6.1. `reference/ar_speculative.py`.
  6.2. `eval/wallclock.py` wires in all lanes; `eval/pareto.py` renders a
  five-lane frontier.
  — **Exit criterion:** v1 figure renders with strict / lossy-oracle /
  best acceptance predictor / best mstar predictor / AR reference, with
  bootstrap CIs.

**Explicitly out of scope until user asks:** Bernoulli commit rule, hybrid
fallback (predictor → small-model → XL), adaptive gamma/T budgeting,
controller/router/retrigger experiments, additional predictor architectures
beyond the four in Phases 3–5.

---

## I. Unit and integration tests

All tests are fast (no HF downloads) except the smoke and consistency tests,
which load the real models once.

### `tests/test_accept.py` — acceptance logic

1. **`test_leviathan_bernoulli`**: fixed `q, p, U` arrays; assert
   `per_position_accept` returns the expected `accepted_j` per the
   published rule. Tests both `accepted_j = 1` when `U < min(1,p/q)` and
   `accepted_j = 0` otherwise.
2. **`test_greedy_argmax`**: protocol with `temperature=0`; assert accept iff
   `argmax(target) == draft_token`.
3. **`test_survival_propagates_rejection`**: craft a case where position 2
   rejects; assert `survived_j = [1,1,1,0,0,...]` and `accepted_j[j>=3]` is
   zero in the returned record (even if the rule technically still computes
   them, the semantic is "did not survive").
4. **`test_min_pq_clamp`**: assert `q=0` and `p=0` do not divide-by-zero;
   both get the `1e-10` floor.
5. **`test_q_mode_provenance`**: different `protocol.q_mode` values yield
   different `q_j` for the same drafter output; assert the record's stamped
   `q_mode` matches.

### `tests/test_commit.py` — commit-length logic

1. **`test_commit_strict_all_accept`**: `accepted_j = [1]*gamma` →
   `L = gamma`.
2. **`test_commit_strict_first_reject`**: first-position reject → `L = 0`.
3. **`test_commit_strict_mid_reject`**: reject at position k → `L = k`.
4. **`test_commit_threshold_equivalent_to_strict_when_binary`**: when
   `alpha_j ∈ {0,1}` and `tau = 0.5`, `commit_threshold == commit_strict`.
5. **`test_commit_threshold_monotone_in_tau`**: `L(tau2) <= L(tau1)` for
   `tau2 > tau1` on the same `alpha`.

### `tests/test_schema.py` — RoundRecord and logging schema consistency

1. **`test_roundtrip_save_load`**: build a `RoundRecord`, save, load; assert
   bit-exact round trip for scalars and `torch.equal` for tensors.
2. **`test_schema_version_mismatch_raises`**: write with version `1`, load
   with `ProtocolConfig(schema_version=2)`; assert `AssertionError`.
3. **`test_protocol_mismatch_raises`**: write with `q_mode=A`, load expecting
   `q_mode=B`; assert the loader refuses.
4. **`test_per_position_arrays_are_length_gamma`**: all of `q_j, p_j, min_pq_j,
   accepted_j, survived_j` have length exactly `gamma`.
5. **`test_no_duplicate_accept_implementation`** *(meta-test)*: grep the
   `accpre/` tree for the Leviathan rule pattern (`min(1.0, p_val/q_val)` or
   `min(1, p/q)`); assert it appears in `accpre/core/accept.py` only. This is
   an AST-level guard, not a string match — it catches silent divergence.

### `tests/test_smoke_pipeline.py` — end-to-end pipeline smoke

1. **`test_strict_specdiff_10_tokens`**: load both models (slow),
   `draft_verify_round` twice for gamma=8, T=2 on a fixed prefix, seed=42.
   Assert `0 <= L <= gamma`; `len(record.accepted_j) == gamma`; `record.L`
   consistent with `accepted_j`; `len(record.U_j) == gamma`;
   `record.round_rng_seed` is set.
2. **`test_strict_baseline_generates_max_new_tokens`**: 64-token run;
   assert `len(generated) - prefix_len == 64`.
3. **`test_oracle_q2_threshold_baseline_runs`**: given a list of strict
   records, evaluate `commit_threshold(record.min_pq_j, τ)` for a sweep
   of τ ∈ {0.3, 0.5, 0.7, 0.9}; assert returned `L_τ ∈ [0, gamma]` for
   every τ and record; assert `L_τ` is non-increasing in τ.
4. **`test_wallclock_measures_only_decode_loop`**: run with a pre-loaded
   pipeline; `elapsed` excludes model load time (use a sentinel clock).

### `tests/test_faithfulness.py` — Controlled Faithfulness@1

Targets the primary v1 quality metric.

1. **`test_cf1_is_1_when_method_equals_strict`**: with
   `method_commit_fn = lambda r: r.L`, assert `CF@1 == 1.0` on an arbitrary
   set of records. Sanity check.
2. **`test_cf1_is_0_when_method_always_shifts_L_by_1`**: with
   `method_commit_fn = lambda r: min(r.L + 1, r.gamma)`, assert every round
   with `r.L < r.gamma` gives `CF@1(round) = 0`.
3. **`test_cf1_equals_L_agreement_under_shared_fallback`**: for a toy
   records set plus a toy `method_commit_fn`, assert
   `CF@1 == mean(1[method_commit_fn(r) == r.L])`. This is the algebraic
   equivalence declared in residual-question 3.
4. **`test_cf1_monotone_in_τ_for_oracle_q2`**: for the oracle-Q2 method at
   τ ∈ {0.3, 0.5, 0.7, 0.9}, assert `CF@1(τ)` is monotone non-decreasing
   as τ increases (a higher threshold commits less and so more often agrees
   with strict's eager commits… or vice-versa; the test pins whichever
   direction the math actually gives and flags drift). Specifically: at
   τ = 1.0, the method commits only positions with `min(1, p/q) ≥ 1`,
   i.e. `p ≥ q` positions, which is a strict subset of strict accepts —
   so `CF@1(τ=1.0) = P(L_strict == L_τ=1.0)` should match the "only-safe-
   commits" ground truth. At τ = 0, the method commits everything up to
   the first `min(1,p/q) < 0`, which never happens, so `L_τ=0 = γ` always
   and `CF@1(τ=0) = P(L_strict == γ)`.

### `tests/test_matched_randomness.py` — RNG reproducibility

1. **`test_round_rng_seed_is_deterministic`**: `derive_seed(protocol, 0, 0)`
   is stable across process restarts; also
   `derive_seed(..., prompt_idx=i, round_idx=j)` is unique per (i, j).
2. **`test_u_j_replays_from_seed`**: given a stored `round_rng_seed`,
   reconstruct the same `U_j` vector. Assert element-wise equality.
3. **`test_strict_is_idempotent_under_fixed_seed`**: run `draft_verify_round`
   twice with the same `round_rng_seed`; assert identical `accepted_j`, `L`,
   `bonus_or_fallback_token`.

### `tests/test_train_infer_consistency.py` — the anti-drift test

*Deferred to Phase 4.* Phase 1 does not build any training path, so there
is nothing to diverge from inference yet. The test is reserved and will
fire the moment the joint-probe predictor lands in Phase 4.

---

**Phase 1 test coverage summary:** `test_accept`, `test_commit`,
`test_schema`, `test_smoke_pipeline`, `test_faithfulness`,
`test_matched_randomness`. These six files must exist and pass before any
predictor code is written.

---

## Summary of design commitments

- **Frozen v1 protocol:** `temperature = 1.0`, `q_mode = "A"`, wall-clock
  throughput, CF@1 as primary quality axis, XL-audit auxiliary only,
  oracle-Q2 threshold as the strong reference baseline.
- **One** draft-verify round function, **one** accept function, **one**
  commit-strict function, **one** `xl_audit`, **one** `ProtocolConfig`,
  **one** `RoundRecord` schema. The Phase-4 anti-drift test enforces this
  against the joint predictor path; meta-tests in Phase 1 enforce it
  against casual duplication.
- Three clearly separated eval layers (L1 / L2 / L3). They never share a
  plot or a headline number.
- Every `RoundRecord` logs `U_j` and `round_rng_seed` so CF@1 is computable
  offline with matched randomness.
- AR speculative decoding is isolated to `reference/`; on a separate panel.
- Bernoulli rule, hybrids, adaptive budgeting, routers — **not in v1**.
- Predictors — **not in Phase 1**.

Ready for Phase 1 as specified in the appended implementation plan.

---

## Phase 1 Implementation Plan (detailed)

Phase 1 delivers the strict SpecDiff baseline, the oracle-Q2 threshold
baseline, canonical logging, canonical wall-clock measurement, and CF@1.
**No predictor code.** XL-audit exists only as an auxiliary side-table
metric.

### P1.1 Files to copy directly from the trusted old project

Copy with only package-path rewrites (no logic changes):

| Old path | New path |
|---|---|
| `HPC_SpecDiff/drafter/mdlm_drafter.py` | `accpre/models/drafter_mdlm.py` |
| `HPC_SpecDiff/verifier/gpt2xl_verifier.py` | `accpre/models/verifier_gpt2.py` |
| `HPC_SpecDiff/data/prompts.py` | `accpre/data/prompts.py` |

From `HPC_SpecDiff/research/lossy_decode.py`, copy **only** the
`threshold_acceptance` function body into `accpre/core/commit.py` as
`commit_threshold`. Leave `bernoulli_acceptance` and `threshold_sweep`
behind (v2 / folded into eval).

### P1.2 Files used only as references (do **not** copy)

| Old path | What to mine it for |
|---|---|
| `HPC_SpecDiff/pipeline/specdiff_pipeline.py` | Decode outer-loop structure; `_accept_reject` and `_sample_adjusted` shapes. Rebuild accept/commit/fallback cleanly — do not copy the class. |
| `HPC_SpecDiff/research/collect_data.py` | Fields logged per round (templates the `RoundRecord`). The two-pass "logits vs hidden states" trick (only relevant if a later phase adds features; Phase 1 does not log hidden states). |
| `HPC_SpecDiff/scripts/run_unified_wallclock.py` | Wall-clock timer placement (around decode loop only). |
| `HPC_SpecDiff/pipeline/aligned_baseline.py` | AR baseline pattern (Phase 6, not now). |
| `HPC_SpecDiff/research/unified_predictor_eval.py` | Per-round aligned eval pattern (Phase 3+). |

### P1.3 Files to create (in order)

Create in this order so each file's tests can pass before the next one
starts.

1. `accpre/core/protocol.py` — `ProtocolConfig` dataclass (§B).
2. `accpre/core/schema.py` — `RoundRecord` dataclass + `save_records` +
   `load_records` (with protocol / schema-version assertions).
3. `accpre/core/accept.py` — `per_position_accept(...)` returning
   `(accepted_j, survived_j, q_j, p_j, min_pq_j, U_j)`. Branches on
   `protocol.temperature` (Bernoulli in v1, argmax only for tests).
4. `accpre/core/commit.py` — `commit_strict` + `commit_threshold`. Also
   one small helper `sample_fallback_or_bonus(...)` shared by strict and
   oracle-Q2 paths.
5. `accpre/core/draft_verify.py` — `draft_verify_round(prefix_ids, drafter,
   verifier, gamma, T, protocol, round_rng_seed) → RoundRecord`. This is
   the *only* place the four core steps are composed.
6. `accpre/models/drafter_mdlm.py` — copied + adapted (see P1.1). Phase 1
   uses only `draft(...)`; `draft_with_features(...)` is a stub (raises
   `NotImplementedError("Phase 2")`).
7. `accpre/models/verifier_gpt2.py` — copied + adapted. Phase 1 uses only
   `score(...)`; `prefix_features(...)` is a stub.
8. `accpre/data/prompts.py` — copied verbatim.
9. `accpre/data/splits.py` — canonical train/val/test split of prompts,
   fixed seed. Phase 1 only needs the test split for baseline runs.
10. `accpre/eval/quality.py` — single `xl_audit(generated_ids, prefix_len,
    verifier) → float`. Marked auxiliary in docstring.
11. `accpre/eval/faithfulness.py` — `controlled_faithfulness_at_1(records,
    method_commit_fn) → {per_prompt: list[float], aggregate: float,
    bootstrap_ci: tuple}`.
12. `accpre/eval/wallclock.py` — `run_lane(lane_name, prompts, max_new,
    protocol)` for lanes `strict` and `oracle_q2_threshold(τ)`. Timer wraps
    the decode loop only. Returns `(records, wallclock_tok_s_per_prompt)`.
13. `accpre/eval/pareto.py` — aggregator that plots
    `(wallclock_tok_s, CF@1)` per lane with per-prompt bootstrap CIs, and a
    separate table/panel for XL-audit.
14. `scripts/run_phase1.sh` — thin shim: load prompts → run strict lane →
    run oracle-Q2 lane at τ-sweep → compute CF@1 and XL-audit → write
    Pareto figure and side table.

Plus `configs/protocol.yaml` with the frozen v1 protocol values.

### P1.4 Minimal strict SpecDiff end-to-end path

```
load protocol ← configs/protocol.yaml
load test prompts ← data/splits.py::test_split()

strict_records = []
for prompt_idx, (prefix_ids, _) in enumerate(test_prompts):
    running = prefix_ids.clone()
    round_idx = 0
    timer.start()
    while len(running) - len(prefix_ids) < MAX_NEW_TOKENS:
        seed = derive_seed(protocol, prompt_idx, round_idx)
        record = draft_verify_round(running, drafter, verifier,
                                    gamma, T, protocol, seed)
        running = cat(running, record.draft_tokens[:record.L],
                      [record.bonus_or_fallback_token])
        strict_records.append(record)
        round_idx += 1
    elapsed = timer.stop()
    tok_s_strict[prompt_idx] = (len(running) - len(prefix_ids)) / elapsed

save_records(strict_records, "outputs/phase1/strict_records.pt")
```

The oracle-Q2 threshold baseline is evaluated **on the same stored
records**: `method_commit_fn = lambda r: commit_threshold(r.min_pq_j, τ)`.
Its own wall-clock lane (P1.3 item 12, lane `oracle_q2_threshold(τ)`) is
a parallel run that actually executes the commit rule online; it is
expected to be *slower or equal* to strict since it still runs the full
verifier. Both lanes land on the Pareto plot; oracle-Q2 serves as a
quality-vs-throughput reference, not a speedup claim.

### P1.5 Canonical logging schema (`RoundRecord`, v1)

All fields required. `protocol` is the full `ProtocolConfig`. Any loader
that finds a record without these fields fails loudly.

```
@dataclass
class RoundRecord:
    # Provenance
    schema_version: int              # == protocol.schema_version
    protocol: ProtocolConfig         # frozen dataclass
    prompt_idx: int
    round_idx: int
    round_rng_seed: int

    # Action
    gamma: int
    T: int
    prefix_len: int                  # len(prefix as seen by this round)

    # Per-position arrays (length gamma, in order)
    draft_tokens: list[int]
    q_j: list[float]
    p_j: list[float]
    min_pq_j: list[float]            # == Q2_j in v1 terminology
    U_j: list[float]                 # Bernoulli draws used by strict accept
    accepted_j: list[int]            # 0/1
    survived_j: list[int]            # 0/1

    # Commit outputs
    L: int                           # commit_strict(accepted_j)
    bonus_or_fallback_token: int     # the single extra token appended
                                     # per round (bonus if L==gamma, else fallback)

    # Timing (per-round)
    draft_time: float
    verify_time: float

    # (Phase 2+) Optional feature fields
    verifier_hidden: Optional[Tensor] = None
    drafter_hidden: Optional[Tensor] = None
    prefix_tail: Optional[Tensor] = None
    verifier_entropy: Optional[float] = None
    verifier_margin: Optional[float] = None
    verifier_top1_prob: Optional[float] = None
```

Phase 1 does not populate the Phase-2+ optional fields. The dataclass has
them as `None` so the Phase-2 extension does not bump `schema_version`.

### P1.6 Exact tests to add before any predictor code

All must pass before Phase 2 starts. Detailed assertions are in §I; this is
the file manifest.

| Test file | What it covers |
|---|---|
| `tests/test_accept.py` | Leviathan Bernoulli at T=1.0, argmax at T=0 (unit-test-only), survival propagation, `min(1,p/q)` clamping, `q_mode=A` provenance. |
| `tests/test_commit.py` | `commit_strict` edge cases, `commit_threshold` τ-monotonicity, equivalence when alpha is binary. |
| `tests/test_schema.py` | `RoundRecord` round-trip, schema-version mismatch raises, protocol mismatch raises, per-position arrays are length γ, **meta-test** that the Leviathan accept pattern appears only in `accpre/core/accept.py`. |
| `tests/test_smoke_pipeline.py` | Strict SpecDiff generates full `max_new_tokens`; oracle-Q2 runs over a τ-sweep with valid ranges; wall-clock timer excludes model load. |
| `tests/test_faithfulness.py` | CF@1 = 1 for identity method, CF@1 = 0 for always-shift method, shared-fallback equivalence to L-agreement, τ-behavior at extreme thresholds. |
| `tests/test_matched_randomness.py` | `derive_seed` is deterministic and unique; `U_j` replays from seed; strict round is idempotent under fixed seed. |

`tests/test_train_infer_consistency.py` is **not** in Phase 1 (no training
path exists yet). It is scheduled for Phase 4.

### P1.7 Phase 1 exit criteria (re-stated)

Phase 1 is done when **all six** hold:

1. Strict SpecDiff runs end-to-end and produces `RoundRecord`s with the
   full P1.5 schema.
2. Oracle-Q2 threshold baseline runs end-to-end, both offline (from stored
   records) and online (its own decode loop), over a τ-sweep.
3. Canonical logging (`save_records` / `load_records`) exists with
   schema-version and protocol assertions.
4. Canonical wall-clock measurement exists: a single timer pattern in
   `eval/wallclock.py`, used identically by every lane.
5. `eval/faithfulness.py::controlled_faithfulness_at_1` exists and produces
   per-prompt + aggregate CF@1 with bootstrap CI.
6. `eval/quality.py::xl_audit` exists and is invoked only for a
   clearly-labeled auxiliary side table; it never appears on the Pareto
   axes.

If any of these fail, Phase 1 is not done and no predictor code may be
written.

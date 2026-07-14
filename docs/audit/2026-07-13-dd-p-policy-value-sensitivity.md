# 2026-07-13 — DD-p value sensitivity: the deployed policy at streaks 0-2 (r2#6 close-out)

Queued by `2026-07-12-dd-leg-calibration.md` (review r2#6): the 7/06
"strategy ≈ wash" replay consumed the profiles' realized DD-leg outcomes,
which sit ~+1pp from stated on the backtest basis — but live 2026 DD legs
realize 0.595 vs 0.734 stated (n=42, exact tail p=0.035). If that shortfall
is real, does the deployed policy's blanket low-streak doubling become
value-negative at streaks 0-2, and would gating it ("surgery") help?

**Verdict: monitor-not-surgery AFFIRMED at the policy level.** The specific
intervention r2#6 contemplated — stop doubling at streaks 0-2 — has lower
expected value than the deployed policy in **all 20 tested season×haircut
cells**, and it is the wrong shape of response even under large haircuts
(the re-solved optimum at high Δ singles only at streak 2 in weak bins,
worth ~2×10⁻⁴ reach-20 probability — trivial). The blanket
doubling-vs-singles contrast is approximately tied at the Δ=0.10 cell and
inverted by Δ=0.139 on mean-max — but the two evidence references for the
true Δ (backtest +0.98±1.1pp; live +13.9±7.5pp at n=42) have to move a long
way before that zone is established. No policy change now; re-measure at the
checkpoint below.

Analysis: `scripts/audit/dd_p_policy_value_sensitivity.py` (tests
`tests/scripts/test_dd_p_value_sensitivity.py`; fast suite 1849→1865).
Artifact with full grids + input fingerprints:
`docs/audit/2026-07-13-dd-p-policy-value-sensitivity.json` (120 profile
files; policy `66d154717ae5…` = the sha stamped on production picks; git
head recorded in the artifact). Two adversarial review rounds (gpt-5.6-sol
xhigh, full-repo read): r1 = 7 findings (1 blocker, 4 major, 2 minor), all
adopted, dispositions inline as (r1#N); r2 fix-verification = 6/7 FIXED with
all 54 diagnostic artifact numbers independently reproduced, finding-1
partial (a residual "scaling law" overclaim, removed) + 3 minor
diagnostic/provenance defects (per-file permutation rng, day-pooled rate,
artifact git_head ordering) — all adopted; converged.

## Method (two layers + a diagnostic, all on the estimated_pa basis)

Environment: the 120 pooled `estimated_pa` profiles (24 seeds × 5 seasons,
rank-1 hit ~0.764 — the realistic basis per the CLAUDE.md PROFILE BASIS
warning), paired under the production different-game DD rule (0.46%
same-game fallback days). Sensitivity parameter: an additive haircut Δ on
the conditional DD-leg rate, `p_both' = max(0, p_both − Δ·p_hit)` — the leg
hits Δ pp less often than the profiles say, uniformly. Primaries untouched
(they measured calibrated, 7/12 doc). **Δ is a specified uniform stress
scenario, not a direct operationalization of the live estimate** (r1#5): the
live 13.9pp is a marginal shortfall on played legs; the
conditional-vs-marginal and pooled-vs-doubled-days denominator gaps are
small here (deployed doubled-day marginal leg rate 0.740 vs pooled 0.7465),
but a shortfall *concentrated* by p2/bin/month/lineup state is out of scope
at n=42.

- **L1 (exact, iid day-type world):** backward-induction value functions for
  reach-K currencies K ∈ {57, 30, 20} on (env-quintile × deployed-bin) day
  types; the deployed table (via its own saved boundaries, as production
  `lookup_action` digitizes) evaluated as a fixed policy vs re-solved optima
  and reference policies; local leg-rate breakevens
  r* = (EV[s+1]−EV[0])/(EV[s+2]−EV[0]) under the deployed continuation — a
  one-step policy-improvement test (exactly the correct continuation for a
  one-step deviation; r1 verified). Solver/evaluator pinned test-for-test
  against `solve_mdp` / `evaluate_mdp_policy`.
- **L2 (realized replay, decision-grade):** the 7/06 comparator's
  different-game replay extended with stochastic leg thinning — each rep
  flips `top2_hit` 1→0 with uniform q = Δ/r̄ (r̄ = 0.7465), so the aggregate
  leg rate drops by exactly Δ; thinned outcomes shared across policies
  (common random numbers, 200 reps). At Δ=0 the replay reproduces the 7/06
  comparator numbers exactly (asserted in-run). This is **comparator
  parity, not production parity** (r1#6): the partnerless-double fallback
  advances +2 on rank-1's own outcome where production's `decide_action`
  demotes to a single (0.46% of days; correcting it moved mean-max <0.001),
  and the fixed 180-position season clock drops 312 of 21,888 paired rows.
- **Diagnostic:** realized-vs-null run structure of rank-1 outcomes (finding
  5), with a within-file day-order permutation null as the load-bearing
  comparison.

## Findings

**1. The blanket low-streak DD gate loses expected value in every tested
cell.** "Deployed but never double at streaks 0-2" has a lower mean
max-streak than the deployed policy in all 20 season×Δ cells
(Δ ∈ {0, 0.05, 0.10, 0.139} × 5 seasons; pooled gap −1.5 to −1.7, profile SE
~0.2). This is lower *expected value*, not strict dominance (r1#4):
trajectory-level at Δ=0 the gate wins 10/120, ties 60, loses 50. L1 agrees
in every currency across the whole grid to Δ=0.20. Mechanically: the
deployed policy spends most of its life at low streaks, where a double's
downside is smallest — gating exactly there removes most of doubling's
upside while keeping it at mid-streaks where legs cost more. The narrower
intervention the model itself would take at high Δ — single at streak 2 in
the weakest bins from Δ≈0.115 — is worth ≈2.3×10⁻⁴ reach-20 probability (the
one-step deviation gain at s=2, d=74, Δ=0.139; independently recomputed in
review): directionally real, value-trivial. If the leg shortfall is ever
established at that scale, the response is a re-solve, not a gate.

**2. Optimal-action flips at streaks 1-2 exist but are value-trivial.**
Re-solving on the shaded environment: streak 0 never stops doubling on any
grid point (Δ ≤ 0.20, all currencies) — r2#6's "streak-0 DDs are
structurally low-risk" confirmed. Streak 2 first flips at Δ ≈ 0.115 and
streak 1 at Δ ≈ 0.185-0.20, in one env bin (the [0.785, 0.800) quintile,
whose empirical legs are weakest). Local leg breakevens under the deployed
continuation at the live horizon (d=74): r* ≈ 0.48 (s=0), 0.50 (s=1),
0.65 (s=2) vs a shaded aggregate leg rate of 0.608 at Δ=0.139 — so at the
live point estimate a one-step deviation to single at s=2 improves the
deployed policy, by the ≈2.3×10⁻⁴ above. (r* is a ratio of near-zero value
differences far from the horizon; read it at d=74, not d=180.)

**3. The blanket contrast: tied at the 0.10 cell, inverted by 0.139 — in one
currency, on the tested grid.** Realized-replay mean-max gap
(always_single − deployed): −1.79 (Δ=0) → −0.98 (0.05) → **−0.07±0.36**
(0.10; 3/5 seasons already positive) → **+0.79±0.34** (0.139; same 3/5).
The profile SEs understate shared-date noise (the 24 seeds re-use each
season's dates), so read this as a band: the pooled crossover sits somewhere
in ≈0.10-0.14, with per-season crossings ranging from ≈0 to beyond 0.139
(r1#3). Reach-20 never inverts on the tested grid: 19.4% (deployed) vs 7.5%
(singles) even at Δ=0.139 — doubling remains how 20 gets reached at all on
this basis, at ~18 extra resets/season. Always-double degrades fastest
(−1.89 vs deployed at Δ=0.139, ≤0 in every season): the 6/10 "always-double
beats the MDP" observation does not survive legs shaded ≥5pp.

**4. What the evidence says about the true Δ: two references, kept
separate (r1#2).** The backtest reference (21,787 simulated legs, exact prod
rule) puts the intrinsic estimated_pa-basis gap at **+0.98pp** (date-clustered
SE ~1.1pp) — but it conditions on realized participation and cannot see
production-only serving effects (7/12 doc, finding 3). The live season reads
**+13.9pp** (n=42, SE ~7.5pp) and includes exactly those effects. These are
estimates of overlapping but not identical quantities; an inverse-variance
blend (~1.2pp) would put 98% of the weight on the backtest and thereby
assume away the production-path effects under investigation — not done.
Decision posture: at Δ ≤ 0.05 the deployed policy is clearly right; the
contested zone starts around 0.10; the live point estimate reaches it but at
n=42/~2σ does not establish it, and the backtest reference is nowhere near
it. **Re-measure, don't re-policy.** Note the 7/12 monitor does NOT
auto-detect this zone (r1#3): the DD-band bucket ([0.70,0.75)×DD-only,
30-day lookback so n≲30) WARNs at 15pp — a true ~10pp gap converges *below*
the WARN bar and would show only as chronic 8-12pp INFO-level readings in
EOD logs. Tripwire: at the 7/12 doc's own accumulation checkpoint (~40 more
legs, season-to-date n≈80-90), recompute the season-to-date leg gap over
pick files (the 7/12 measurement); if it holds ≥10pp, rerun this script and
take the re-solve question seriously.

**5. Realized rank-1 sequences show strong long-window suppression —
exploratory but robust to the order-nulls tested; its effect on policy
values is direction-dependent, NOT a uniform inflation (r1#1, corrected from
this doc's own first draft).** Measured with no replay machinery
(`run_structure_diagnostics`, all numbers in the artifact): all-hit
20-windows per profile **0.108 observed vs 1.037 expected iid** at the
file's own rate (×0.10), and **×0.14 under a within-file day-order
permutation null** — the permutation null fixes each file's rate, hit
count, and length, so between-file/season rate heterogeneity cannot explain
it (within-file temporal rate variation survives the null and is one of the
candidate mechanisms below; review r2). Suppressed
in **all 5 seasons** (observed/iid: 2021 ×0.23, 2022 ×0.03 [1 vs 38.5],
2023 ×0.26, 2024 ×0.00 [0 vs 34.2], 2025 ×0.00 [0 vs 3.3]). Run-length
survivor vs permutation: ≈1.05-1.12 through run 8, ×0.90 (10), ×0.93 (12),
×0.74 (15), **×0.28 (20)** — long-window suppression with a thin measured
tail (63 runs ≥15, 6 ≥20); the small-L ratios are non-monotone, so this is
NOT a demonstrated monotone hazard. Lag-1 autocorrelation is +0.03 and
stated-p windows are, if anything, favorably clustered
(independent-given-p1 expects 1.155 windows) — pairwise statistics and
between-unit rate differences don't reach it. Effective sample ≈ the 5
seasons, not 120 files (the 24 seeds' sequences are heavily correlated
within a season). Consequences for policy values, measured not assumed: the
realized-vs-iid gap is policy-dependent with no clean one-dimensional law
(r2#1) — iid **understates** deployed (23.1% iid vs 31.7% realized reach-20
at Δ=0) and always-double (31.7% vs 40.8%), is ≈exact for the no-DD-low
variant (19.8% vs 20.0%), and **inflates always-single ×3.6** (26.8% vs
7.5%); run length is suggestive (always-single needs the longest
consecutive-hit calendar runs) but does not explain no-DD-low. So on
realized data doubling's edge over singles is *larger* than any iid solve
suggests, P(57)-scale values stay phantom for every policy, and per-policy
realized replay is the only evaluator trusted here — L1 is structural only,
L2 is the decision layer. Mechanism unidentified:
candidates include schedule/phase structure, conditional miscalibration
along runs, and repeat-batter form regression (the model's recency features
chasing hot batters whose true rate regresses); none tested. Open thread:
condition run survival on repeat-batter identity.

**6. The re-solve arm is an optimistically-biased in-sample benchmark, not a
decision (r1#7).** A policy re-solved on the shaded environment beats
deployed on realized replay at every Δ (+0.72 to +2.78 mean-max) — but its
bins are fit on the same 120 profiles it is replayed on, it is optimized for
P(57) yet graded on mean-max, and its Δ=0 per-season spread (+4.1 to −3.9)
shows the optimism plainly. This does NOT reopen the 6/10 settled decision
against deploying the estpa re-solve; it only says a re-solve is the shape
of the response IF the finding-4 checkpoint ever establishes a large Δ.

## Caveats

- Δ is a uniform stress scenario (see Method); a concentrated shortfall is
  out of scope. The carried per-day `p2` is not used in shading/thinning.
- L2 is comparator-parity, not production-exact (fallback + fixed season
  clock; both quantified above and immaterial to the headline gaps).
- The 24 seeds re-use the same ~908 dates (~7.4×); profile-level SEs
  understate date-cluster noise (7/12 doc r2#5) — per-season sign tables
  carry the robustness weight, and the effective n for finding 5 is closer
  to 5 than 120.
- estimated_pa conditions on realized participation (7/12 doc, finding 3).
- L1 optima and the re-solve arm condition on env-quintile bins computed
  from the full pool (in-sample); the deployed policy is evaluated through
  its own saved boundaries, as production digitizes confidences.

## Reproduce

```
uv run pytest tests/scripts/test_dd_p_value_sensitivity.py -q
uv run python scripts/audit/dd_p_policy_value_sensitivity.py --reps 200
```

Runtime ~6 min locally; deterministic (fixed thinning/permutation seed;
Δ=0 anchor asserted against the 7/06 comparator).

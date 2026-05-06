# MDP Objective Audit (2026-05-06)

**Status:** Phase 1 audit. No solver changes proposed in this document.
**Owner thread:** `bts-motor-session` (BTS bus #254 → #257).
**Companion work (Codex):** alternate-policy evaluation plumbing for `bts simulate exact --policy-file` and the falsification harness.

## 1. Why this memo exists

Bus #254 proposed a CVaR-α MDP as a "winning" policy lever, on the premise that the current MDP maximizes E[streak_length]. Bus #255 (Codex) refuted that premise by reading the solver: the current `solve_mdp` is a reachability backward-induction whose value function at every state is exactly P(reach streak 57 | state). The CVaR-over-streak-length objective therefore solves a *different* problem and is not, on its face, a P(57)-improving change. This memo documents the current objective formally, names the candidate alternates honestly, and concludes with a Phase-2 decision.

## 2. The current objective, formally

### 2.1 State and dynamics

`src/bts/simulate/mdp.py:115-213`. State:

```
S = (s, d, saver, q) ∈ {0,…,57} × {0,…,season_length} × {0,1} × {0,…,n_bins-1}
```

with `s` = current streak length, `d` = days remaining, `saver` = streak-saver-still-available flag, `q` = today's quality bin index. Actions A = {skip, single, double}. Per-day transitions are read from the empirical bin manifold (`p_hit[q]`, `p_both[q]`, `frequency[q]`); see lines 138-150 for early/late-phase splitting.

### 2.2 Reward and value function

The recursion (lines 156-202) is:

```
V[57, *, *, *]      = 1.0                                 (line 157, terminal indicator)
V[s, 0, *, *]       = 0.0  for s < 57                      (line 153, zeros initialization)
V[s, d, saver, q]   = max_a Σ_{q'} freq[q'] · V[next(s,a), d-1, saver(s,a), q']    (lines 167-202)
```

with `next(s, a)` and `saver(s, a)` reading off the existing transition rules including the streak-saver carve-out at `10 ≤ s ≤ 15`.

### 2.3 What this value function represents

Because the only positive reward source is the terminal `V[57, *, *, *] = 1.0` indicator and `V[s<57, 0, *, *] = 0` is the only other boundary, induction gives:

```
V[s, d, saver, q] = P(reach streak 57 within remaining d days | start at (s, d, saver, q), follow optimal policy)
```

The argmax at line 200 selects, at each state, the action that maximizes this reachability probability. The reported `optimal_p57` (line 205) is the expectation of this reachability over today's bin frequencies starting from `(s=0, d=season_length, saver=1)`. **The current MDP is therefore a P(57)-optimizing reachability DP under the assumed transition model.**

### 2.4 What this objective is *not*

- Not E[streak length].
- Not E[number of streaks of length ≥ k] for any k < 57.
- Not a discounted-sum-of-rewards problem (there is no discounting; it is reachability with hard horizon `season_length`).
- Not robust to estimation error in the transition model — `p_hit`, `p_both`, `frequency` enter as point estimates, not ambiguity sets.

## 3. Tests-as-documentation (companion: `tests/simulate/test_mdp.py`)

Three new tests harden this specification. All three live in the same file as existing reachability checks (`test_terminal_state_value_is_1`, `test_zero_days_value_is_0`, `test_optimal_beats_or_matches_always_single`).

1. `test_value_equals_closed_form_reachability_on_tiny_mdp` — picks a hand-computable state where V can be derived analytically from the recursion and asserts numeric equality within float tolerance. Covers the load-bearing claim of section 2.3.
2. `test_optimal_p57_matches_initial_state_expectation` — asserts `solution.optimal_p57 ≈ Σ_q freq[q] · V[0, season_length, 1, q]`, locking in the contract of line 205.
3. `test_value_function_is_probability_in_unit_interval` — every entry of `V` lies in [0, 1] (a necessary condition for any reachability semantic; would catch a future objective change that allows V > 1 or negative rewards).

These tests serve two purposes: a regression guard against accidental objective drift, and executable documentation of what the solver is.

## 4. Candidate alternate objectives

Evaluated against three criteria:
- **(A) Mathematical compatibility with P(57) as the win metric.** Does the alternate objective, when optimized, plausibly *increase* P(57) versus the current solve, or only redistribute mass elsewhere?
- **(B) First-principles fit for the BTS streak structure.** Are the assumptions of the alternate objective (loss distribution shape, ambiguity set structure, etc.) actually realized in this problem?
- **(C) Implementation cost vs evidence of benefit.** Is the gap to the current optimum, under defensible modeling assumptions, big enough to justify the work?

### 4.1 CVaR-α over terminal streak length

**Definition:** maximize CVaR_α(streak_length) for some α ∈ (0, 1].

**Verdict: ruled out.** The streak distribution under any production-shaped policy is heavily right-tailed with the win mass concentrated at exactly `streak_length = 57` (and beyond). CVaR_α[streak] for α < 1 *downweights* this tail because the upper tail is where the desired event lives. The objective also conflates "streak length 56" (a loss) with "streak length 57+" (a win) on a continuous scale where they should be a hard indicator. **Wrong topology for the BTS win condition.**

### 4.2 CVaR-α over a different random variable (e.g., per-day P(hit) shortfall)

**Definition:** any CVaR objective formulated over a *non-streak-length* random variable, e.g., the lower-tail of realized per-pick hit rate.

**Verdict: not motivated.** The current MDP already absorbs per-pick uncertainty through the `p_hit` / `p_both` bin estimates. A CVaR objective over per-pick shortfall would be implementing risk-aversion on the *parameters* of the Bernoulli draws, which is more naturally framed as the DR-MDP in section 4.4. Folding that into a CVaR layer adds notation without changing the math.

### 4.3 Expected streak length / lexicographic streak-then-time

**Definition:** maximize E[streak_length], or maximize E[streak_length] subject to P(57) being already maxed.

**Verdict: ruled out for primary objective.** P(57) is the only thing the contest pays. Maximizing E[streak] would trade win probability for prettier-looking 30-streak misses. As a *secondary* objective (lexicographic tiebreak when multiple actions are P(57)-equivalent), it could in principle pick "more conservative" actions among ties — but ties are vanishingly rare on a continuous reachability landscape and would appear only in degenerate states.

### 4.4 Distributionally Robust MDP (DR-MDP) over bin parameters

**Definition:** replace the point estimates `(p_hit[:], p_both[:], freq[:])` with two ambiguity sets, each around a *distinct* statistical object:

- **Per-bin hit-rate ambiguity** `U_hit ⊂ ∏_q [0,1]^2`: an ambiguity set over the per-bin parameter pairs `(p_hit[q], p_both[q])`. Each bin's pair is a per-bin Bernoulli/joint-Bernoulli estimate; per-bin Wilson- or Clopper-Pearson-style intervals are appropriate scaffolding for a first cut, **conditional on** the binning being held fixed and within-bin paired-Bernoulli dependence being either modeled (joint-Bernoulli) or sample-bootstrapped at the day level.
- **Bin-frequency ambiguity** `U_freq ⊂ Δ^{n_bins-1}` (the simplex): an ambiguity set over the next-day bin-distribution vector `freq[:]`. This is a multinomial over `n_bins` cells, not a per-bin scalar; Wilson is the wrong shape. Defensible candidates are a Dirichlet credible region around the empirical frequencies, a multinomial-bootstrap quantile ball, or a profile/day block-bootstrap over the (season, day) → bin assignment empirical distribution that respects within-day correlation between picks.

The split matters because `U_hit` and `U_freq` carry different statistical structure: `U_hit` is per-bin ("how well do we know this bin's hit rate?"), while `U_freq` is over the simplex ("how well do we know which bins we'll see tomorrow?"). Conflating them — e.g., putting a per-bin Wilson around `freq[q]` independently — overstates the frequency uncertainty and ignores the simplex constraint.

The robust recursion is then

```
V_robust[s, d, saver, q] =
    max_a   min_{(p_hit', p_both') ∈ U_hit}
            min_{f' ∈ U_freq}
                Σ_{q'} f'[q'] · V_robust[next(s, a; p_hit'[q], p_both'[q]), d-1, saver(s, a), q']
```

i.e., act optimally against the worst-case joint realization of (per-bin hit/both, next-day bin distribution) within the two ambiguity sets. Whether the inner min should treat `U_hit` and `U_freq` as independent (rectangular) or jointly constrained is itself a design choice — see the Phase-2 recommendation below.

**Verdict: candidate, but evidence-gated.** This *is* a P(57) objective — it hedges against estimation error in the bin manifold without changing the win metric. Its policy can differ from the current point-estimate optimum *only when at least one of the ambiguity sets is large enough to swing the inner min through the argmax at some state*. Whether that ever happens in practice is an empirical question:
- (A) compatible with P(57): yes — robust max-min over P(57) reduces to the current solver when both ambiguity sets are singletons.
- (B) first-principles fit: depends on (i) whether any bin's per-bin paired-Bernoulli sample size is small enough for `U_hit[q]` to be wide; (ii) whether the multinomial dispersion across folds/seasons/seeds makes `U_freq` non-trivial. The 24-seed pool with quartile binning probably has tight `U_hit` per bin; `U_freq` may still be non-trivial because frequency depends on the upstream P(hit) distribution which itself shifts across seeds and seasons.
- (C) cost: O(|U|) extra inner work per state plus a non-trivial design choice for both ambiguity sets. Not free.

**Phase 2 deliverable if pursued (a measurement, not a solver):** quantify the maximum P(57) gap between the current point-estimate optimum and a robust optimum *across a small grid of ambiguity-set constructions* — e.g., (Wilson-on-`U_hit`, Dirichlet-on-`U_freq`), (paired-bootstrap-on-`U_hit`, multinomial-bootstrap-on-`U_freq`), and possibly a (season, day) block-bootstrap over both to respect within-day pick correlation. The ambiguity-set choice is part of the measurement design, not a settled prior. If every construction's max |ΔP(57)| ≤ block-bootstrap CI half-width on the harness, the lever is too small for the cost regardless of which set is "right". If at least one construction shows a defensible gap, that result motivates a follow-up scoping discussion before any production solver code.

### 4.5 Bin-side improvements (not strictly an objective change)

**Definition:** keep the solver, change the inputs. Cross-fitted bin-rate calibration (#2 in tracker), better binning resolution, BOCPD drift detection on the bin manifold (#6), or pooled multi-seed bin estimates that close the seed=42 outlier flagged in `CLAUDE.md`.

**Verdict: this is the most likely real lever, but it's not a Phase-2 of this PR.** If the objective audit concludes the solver is correctly specified, the next leverage is in the bin estimation, not the recursion. Surfaced here so the conclusion section can name it.

### 4.6 Summary table

| Objective                                | (A) P(57) compat | (B) First-principles fit | (C) Cost vs evidence | Verdict     |
|------------------------------------------|------------------|---------------------------|----------------------|-------------|
| CVaR-α over streak length                | wrong topology   | tail mass on win event    | n/a                  | **out**     |
| CVaR-α over per-pick hit shortfall       | redundant w/ 4.4 | n/a                       | n/a                  | **out**     |
| E[streak] / lexicographic                | secondary at best| ties vanishingly rare     | low gain             | **out**     |
| DR-MDP over bin parameters               | yes              | depends on bin tightness  | medium               | **candidate**|
| Bin-side improvements                    | yes (input-side) | direct (data → solver)    | medium-high          | **candidate (separate track)**|

## 5. Conclusion + Phase 2 recommendation

**The current `solve_mdp` is correctly specified for the BTS win metric.** Its value function is reachability probability; its argmax maximizes P(57); its `optimal_p57` is the expectation of that reachability over the initial bin distribution. There is no general-purpose objective swap that improves P(57) without also assuming something the current model doesn't assume (parameter ambiguity, a different metric, etc.).

The single defensible Phase-2 candidate inside the *solver* is **DR-MDP over bin parameters (4.4)**, and it is *evidence-gated* — pursuing it without first measuring the maximum point-vs-robust P(57) gap on the current bin manifold risks shipping a more expensive solver that produces an identical policy. The measurement itself is structurally cheap (construct ambiguity sets, solve robust DP, diff against point-estimate solve, compare to harness CI half-width), but the ambiguity-set construction is a non-trivial design choice — see section 4.4 for the split between `U_hit` (per-bin paired-Bernoulli) and `U_freq` (multinomial over the bin simplex), and the recommendation to evaluate a small grid rather than a single "right" set. That measurement should precede any production code change.

The *most likely* P(57) lever is in **bin-side work (4.5)** — cross-fitted calibration, multi-seed pooling, or drift-aware re-binning — and it lives outside the solver. It is named here only so the audit conclusion does not implicitly endorse the solver as the locus of remaining gain.

### Recommended next motor steps

1. Land this memo + the three reachability-semantics tests (this PR) and Codex's `--policy-file` plumbing for alternate-policy evaluation in the harness.
2. Build a one-script DR-MDP measurement (`scripts/dr_mdp_gap_measure.py`) that evaluates point-vs-robust max(|ΔP(57)|) across a small grid of ambiguity-set constructions — at minimum (Wilson-or-Clopper-Pearson on `U_hit`, Dirichlet on `U_freq`) and (paired-day-bootstrap on `U_hit`, multinomial-bootstrap on `U_freq`); optionally a (season, day) block-bootstrap variant. Report the gap at the initial state for each construction and compare against the harness block-bootstrap CI half-width. If at least one construction's gap exceeds the CI half-width on at least one bin manifold, **then** open a follow-up scoping PR before any production solver change.
3. Independently of the solver track: pick up multi-seed bin pooling as the bin-side P1 work item (mentioned in `CLAUDE.md` as blocking trust in single-seed deltas since 2026-04-14).

Phase 2 of this thread is a *measurement*, not an *implementation*. That ordering forces the evidence to come before the code.

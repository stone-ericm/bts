# C1 rank 4b — design review round 1

## Verdict

**BLOCK.** Reviewed rev 1 at main `12ac1c46790c047fbc2c7bccfcfaa913fa20c7b7`.

Registration SHA-256: `1b3c36acf445b96178e362a1bc7b0a78b1f413b6dd0c6a3b6762718e39c825fc`. Main advanced concurrently to `0fff8f234bdd9b721060bf6cf04bdb0066e03212` during review. Final verification found the registration, named authorities and reviewed code unchanged; the cycle index changed candidate links/statuses only, with its gates unchanged. The tracked worktree remained clean.

The weakest claim is that a positive A2−A0 result would establish the value of changing the objective and furnish the trade for the policy Eric would approve. It need not do either. A1 removes A0's tail, the gate can pass when A2 loses to A1, the proposed P(57) comparison does not fix the policies' boundary mappings or dependence scenarios, and the eventual production artifact is not defined or bound to the table.

The smallest repair is to retain one candidate, make A1 a matched hybrid control, require an objective-specific contrast as well as the deployed contrast, and freeze the replay, projection, provenance and decision-object contracts below. No second objective, bin search or new data campaign is needed. The historical screen also needs to be distinguished from C1's recorded 2027 validation requirement; this review cannot waive that requirement.

Evidence: code, documents and relevant history were inspected read-only. Small Python checks used synthetic inputs only. No profile/artifact under `data/` was read and no historical replay or July parity check was executed. Historical numerical anchors below are carried from the named memos, not independently reproduced measurements. No tracked file was edited. This is the complete important-issue list for round 1; round 2 should verify these dispositions, then freeze or defer under the pace rule.

## Findings

### F1 — P1: the decomposition changes more than its labels say, and can pass with a losing objective

**Locations:** registration lines 21–23, 62, 72–78; plan W4 lines 191 and 198; frozen decision memo lines 71 and 76.

A0 is reach-57 **plus** an E[best] tail. A1 is just a P(57)-optimal re-solve. `solve_mdp` initializes all policies to skip and uses skip-first `argmax` (`src/bts/simulate/mdp.py:240–245,284–288`); when `s+2d<57`, every action has value zero. Thus A1−A0 also removes the tail, and A2−A1 includes restoring it. Neither subtraction isolates the asserted component. The separate-hypothesis constraint is explicit in W4: boundary, transition-rate and objective changes must not obtain acceptance through one bundled A/B.

Even with a repaired A1, the current gate can recommend the objective when it hurts. Synthetic contrasts A1−A0 = +1.00 and A2−A1 = −0.50 in each season yield A2−A0 = +0.50, passing the practical bar and all five season signs; stress and reach-20 could also pass. The decomposition merely prints this failure without preventing acceptance.

**Fix:** A1 must have matched reach-57 routing followed by matched E[best] tail routing, using A2's fitted environment and otherwise identical rules. Require the objective-specific A2−A1 gain and the A2−A0 switch gain separately. Call A1−A0 the entire declared re-solve/adaptation effect, including horizon/bin conventions; do not call it a pure rates contrast. A1 is a diagnostic control, with no independent ship pathway. Edits E2, E5 and E6.

### F2 — P1: “held-out” does not describe all arms or the production acceptance gate

**Locations:** registration lines 10, 34–35, 54–57, 85, 113; approved C1 proposal lines 12–14 and 77.

Holding each season out of **A1/A2's** policy fit is sensible, and grouping all 24 seeds by season avoids seed-level leakage. It is not a chronological backtest: early folds use later seasons to fit policies. More importantly, A0's shipped tail was fitted on **all five 2021–2025 seasons**, including every nominal holdout. This is established by `scripts/rebuild_tail_policy.py:108–115,138–145`, its original introducing commit `0abf503`, and the September tail memo lines 42–48. The base policy's original fitting profiles are lost (same memo lines 88–91). Fixing A0 does not erase its outcome exposure. The July and June results have also already been examined. This can screen a newly frozen fitting procedure; it cannot be presented as an untouched comparative estimate for all arms.

C1's approved text says each candidate is tested on 2027 games not used for fitting. The later 4b addition supplies a trade-table gate but records no express waiver of that condition. Rev 1 contains no 2027 validation or deferral path and describes a positive historical result as a recommendation to switch. D7 is necessary, but it does not silently amend the cycle's validation contract.

**Fix:** label the historical contrast accurately, retain A0 unchanged, disclose its exposure, and state that historical positivity alone does not satisfy the applicable 2027 requirement. Either publish a separately reviewed prospective validation contract before its observations, or obtain a recorded owner amendment before using history alone for acceptance. The latter is not supplied by this review. Edits E1, E3 and E6.

### F3 — P1: five fitted fold policies are not the single artifact whose trade Eric approves

**Locations:** registration lines 34, 63–68, 108–114.

The replay tests five A2 policies. Section 9 then creates an unspecified full-season artifact **after** Eric approves. It does not say which rates, horizon or boundaries that artifact uses, whether it is an all-five-season re-fit, or how its jackpot projection relates to the average of fold projections. An all-five-season fit is a sixth policy. Its value cannot be called held out, and its actions can differ from every fold. Loader equality proves the artifact matches its embedded rates; it does not bind that artifact to the policy trade previously shown to Eric.

**Fix:** freeze the final-fit rule now. Build the reviewable final decision object and calculate its separately labelled projections before D7; bind approval to its manifest and action-table hashes. Keep the fold table labelled as evaluation of the fitting procedure. Packaging after approval may reproduce that decision object, not choose a new policy. Edits E3, E5 and E7.

### F4 — P1: the calendar/availability contract can still produce an asymmetric or compressed-clock green

**Locations:** registration lines 31–32, 40, 49 and 93–96.

“Remaining contest dates” is not an executable clock definition. Production uses an exclusive season-end **calendar-day difference**, then its own table cap (`src/bts/strategy.py:227–239`, `mdp.lookup_action:198–200`); it does not count retained profile rows. `split_by_phase_pooled` instead chooses the last 30 distinct **observed** dates (`pooled_policy.py:217–225`). The precedent's row-stream DP also uses `len(days)-idx` (`scripts/evaluate_mdp_dd_guardrail.py:340–344`), so adopting it alone does not solve gaps.

For synthetic profile dates September 25 and September 28 with exclusive end September 29, row countdown is `[2,1]`, calendar countdown is `[4,1]`. Missing September 26/27 must not grant extra policy days or redefine the late phase. “Absent dates are skipped” currently cannot distinguish an evidenced no-opportunity day from an unobserved playable day. Nothing audits availability on days an arm skips, as the plan explicitly requires. The offline producer selects actual starter-matchup participants and a retained top-N pool (`backtest_blend.py:672–674,715–717`); these are not pre-lock availability witnesses.

There is also a phase-boundary mismatch to freeze: production has one saved base cutpoint array, whereas pooled fitting computes separate early/late quantiles. The June methodology specifically warned against phase-switching A0's boundaries (lines 14–19 and 61–62). An A0 replay through candidate or held-out quintiles changes the comparator.

**Fix:** use an outcome-free calendar manifest with explicit end/counting/no-opportunity conventions; classify missingness separately; apply fixes identically to every arm; pin A0's original boundary mapping and table cap; freeze A1/A2's classifier and rates consistently. A0 with instantaneous trusted replay best is an idealized healthy-artifact comparator, not a reconstruction of 2026 entry/refresh behavior. Edits E2–E4.

### F5 — P1: the jackpot projection lacks both a comparable environment and the required dependence stress

**Locations:** registration lines 63–68, 108 and 111; plan rule 1.5, line 38; decision memo D1, line 30.

The m-state fixed-policy DP is a defensible **conditional model projection**, not a season-frequency estimate. Fitting-rate and held-out-rate versions are useful only when each policy keeps its own frozen classifier. Rev 1 does not require that, does not prevent holdout re-quantiling, and does not say how A0's original cutpoints and A2's fitted cutpoints coexist. Comparing each arm's own fitted rates would mix environment changes into the jackpot cost. The July script already supplies the right starting pattern: a joint environment over environment-bin × deployed-bin (`dd_p_policy_value_sensitivity.py:166–208`); the generic pooled evaluator expressly treats fresh holdout quintiles as the same bin indices (`pooled_policy.py:246–250`), which is unsuitable here.

Subtracting 0.02/0.04 from rates changes marginal calibration, not serial outcome dependence. Rule 1.5 requires both. July finding 5 demonstrates why an iid disclaimer cannot substitute: the milestone error varies by policy, with opposite directions for singles and doubles (memo lines 138–169). A calendar-ordered probability DP preserves probability ordering but still assumes conditional independence; that alone is not a serial-dependence stress either.

**Fix:** estimate a common evaluation environment through frozen classifiers, enforce legal availability in the transitions, and predeclare an actual history-dependent outcome scenario. Report paired jackpot-cost differences and the entire scenario range, with no fitted “true cost,” no selection of a favorable cell and no zero-cost conclusion from unresolved projections. E5 gives one small explicit dependence stress; it is speculative, not an estimated correction. Other stress definitions would need to be frozen before outcomes and verified in round 2.

### F6 — P1: the A1 equality check does not test the evaluator's new m dimension

**Location:** registration line 68.

Agreement with `solve_mdp` is a useful necessary check only when its rates, horizon, phase frequencies and execution kernel agree. A1's reachable-region actions do not depend on m; its hybrid tail has zero P(57) regardless of m. A broken evaluator that replaces best by current streak after every reset can therefore pass that check.

**Verified synthetic counterexample:** target 3, five days, initial `(s,m)=(0,0)`, saver off, `p_hit=p_both=0.5`. Always-double reaches the target with probability **0.59375** with either correct m or the erroneous `m=s`. A policy that doubles for `m<2` and singles for `m>=2` reaches it with **0.50** when best survives resets, versus **0.59375** with the erroneous collapse. This is a demonstration of the proposed validation's blind spot, not a measurement of any BTS arm.

**Fix:** independently enumerated tiny paths for an m-dependent policy, plus saver, phase transition, crossing 57 from 56, legal demotion and A0 routing/stop cases. Also test the full-season phase-aware E[best] solver, which the current tail solver does not implement: `solve_emax_season_best` takes one stationary rates vector (`tail_policy.py:159–198`). Existing late-tail tests do not certify that extension. Edit E5.

### F7 — P2: uncertainty, stress execution and regression dispositions are not frozen tightly enough

**Locations:** registration lines 50, 55–57, 73–83, 88 and 105.

The ordinary across-profile SE treats 120 shared-date trajectories as independent. Merely admitting an effective n near five elsewhere does not make the main table's SE honest. In a synthetic calculation, season effects `[1,1,1,1,-2]`, each repeated 24 times, have mean **0.40**, naive profile SE **0.1100**, versus **0.60** for the five season means. Those effects satisfy the current pooled practical bar and four nonnegative seasons despite one season losing two streaks. The overlapping four-season fits also mean even a five-season SE is descriptive, not a proved sampling CI.

The +0.25, 4-of-5, Δ=0.10 and 2 pp bars are defensible pragmatic screen choices, not demonstrated MDE/power. But the nonnegative sign rule admits four exact ties and one winning season; the reach-20 guardrail has no declared Δ; and a material guardrail breach currently becomes “inconclusive” even with positive mean gain. There is no individual-season regression limit. The closest precedent prespecified a −0.25 season-regression limit (guardrail prereg lines 244–245).

“July haircut” also leaves build choices open: the old L2 re-solves a policy at each Δ, uses 200 thinning replicates and a pooled marginal leg rate (`dd_p_policy_value_sensitivity.py:754–793`). A fixed candidate cannot be credited with robustness from a different re-solve at each stress. Number of replicates, seed, common random numbers, marginal versus conditional thinning, partnerless treatment and Monte Carlo precision need explicit contracts.

**Fix:** season-level reporting, clearly conditional uncertainty; freeze the Δ=0 policies across stresses; specify thinning and its Monte Carlo uncertainty separately; apply guardrails at both Δ=0 and 0.10; classify failed valid-data guardrails as negative. Edit E4–E6. The proposed individual-season cap is an engineering risk limit carried from the precedent, not a measured power result.

### F8 — P2/P3: provenance, consequences and a few claims need exact bindings

**Locations:** registration lines 4, 29, 69, 111, 123–125; plan registration fields, line 198.

The registration promises input hashes at run start but does not bind the retained evidence manifest, original forecast-generation recipe, current evaluator/solver recipe, calendar and exact classifier. “Conditions on realized participation” is a useful limit, not the requested exact information-set description. State-weighted action changes do not specify whose states are visited or whose streak qualifies as ≥10. A comparison of actions at different own states measures trajectory divergence; a comparison at common visited states measures a policy-map difference. Both can be useful, but the table must identify which it shows.

The cycle index actually exists at `docs/sota_audit/2026-10-04-c1-cycle-index.md`, not the `docs/audit/` path in line 4. “The solves take seconds” has not been measured for the proposed phase-aware whole-season solves and added evaluator. “No trajectory ever reached 57” and the X→Y headline should not preselect the new run's observed result or a favorable projection cell.

**Fix:** outcome-free provenance freeze, explicit conditional information set and forecast parity; define both state-visit views; correct the link and label compute sizing as reasoning. Edits E1, E3, E5 and E7.

## Verbatim edits

These replacements apply to the registration, not its frozen authorities. They are proposed text; this review has not applied them. Freeze or defer only after their implementation choices and authority disposition are verified in round 2.

### E1 — header and plain-language claims

Replace lines 4–6 with:

```markdown
**Cycle:** C1 (`docs/sota_audit/2026-10-04-c1-cycle-index.md`), approved by Eric 10/04 with 4b added.
**Owner rulings it serves:** D1 = Option 2, optimizing our own expected season-best streak, capped at 57. Eric receives both the historical streak-value evidence and the explicitly model-dependent jackpot-cost scenarios before D7 approval of a concrete policy. Historical positivity alone does not waive C1's requirement for validation on unfitted 2027 games. A separately reviewed prospective contract must be frozen before its observations, unless Eric records an explicit amendment to that requirement. No change affects picks before independent acceptance and D7 approval.
**Exposure:** row X-31 and the outcome-free provenance manifest are published before any new outcome-bearing computation. This historical screen uses only the frozen 2021–2025 estimated-PA profiles and declared outcome-free calendars. No 2026 outcome is read or fitted (D3 = RESERVE).
```

Replace the plain-language “The test” and “What can't be measured” bullets with:

```markdown
- **The test.** We screen one full-season longest-streak fitting procedure on 2021–2025 realized sequences, holding each season out of its new policy fit. The deployed comparator is a fixed, historically exposed policy; this is not an untouched test of all arms or a chronological deployment simulation.
- **What can't be measured.** Five historical seasons cannot resolve the chance of 57. The July replay observed no reaches of 57; the new run reports its own counts. Jackpot chances and their differences are model projections under registered calibration and dependence scenarios, not measured contest frequencies.
```

### E2 — replace §2

```markdown
## 2. Arms
Every arm carries its own streak s, season best m, saver and calendar state, initialized at (0, 0, available). Streak and best are capped at 57; reaching 57 is absorbing. Every arm uses the same frozen forecast/candidate rows and legal fallback rules.
- **A0, pinned deployed hybrid:** the shipped reach-57 table (`mdp_policy.npz`, sha `66d15471…`) through its own single saved boundaries and horizon cap; when s + 2d_effective < 57, the paired shipped one-bin tail (`dc5d0c99…`) through its own boundaries and stop rule. Full digests are mandatory in the provenance manifest. Replay best is instantaneous, exact and trusted by construction. This represents the healthy-artifact deployed policy with perfect replay-state observation, not actual 2026 entry, delivery or profile-refresh timing.
- **A1, matched hybrid control:** the reach-57 re-solve on exactly A2's fitting-season environment, classifier, horizon and execution assumptions, followed by A2's matched E[best] continuation whenever 57 is unreachable. It retains the same objective-routing predicate as A0. It is a diagnostic control with no separate acceptance or ship pathway.
- **A2, the sole candidate:** the full-season E[best] policy on that identical fitted environment; only its objective routing differs from A1. Both solves use the same saver rules and registered tie handling: reach-57 uses skip-first exact ties; E[best] uses the explicit terminal stop and play-first exact ties.
- **Comparators:** always-single and legal always-double, with the same calendar, saver, cap and availability handling.

A1 − A0 is the combined declared re-solve/adaptation effect, including rate, boundary, horizon and tail-resolution differences. A2 − A1 is the matched objective-routing effect. A2 − A0 is the proposed switch effect. The objective and switch contrasts must both pass §6; a rates/boundaries improvement alone cannot accept the objective change. No alternative objective or bin scheme is tried if A2 fails.
```

### E3 — replace §3

```markdown
## 3. Frozen inputs, rates, classifier and split
- **Inputs and recipe:** before outcomes are decoded, publish a manifest listing the exact 24 seed IDs × five seasons, profile paths and SHA-256s, retained W0 evidence-manifest identity and byte-equality check, profile-generation command/commit and estimated-PA mode, schema, all replay/solver source hashes and reviewed execution commit, dependency lock hash, full A0 artifact digests and pairing, and the calendar manifest. Missing provenance or a byte mismatch stops the screen. Hashing current files alone does not establish equality to the retained evidence.
- **Information set:** forecasts remain frozen. The existing offline estimated-PA surface selects realized starter-matchup participants and lineup slots and retains a ranked top-N pool. It does not reconstruct pre-lock eligibility, scratches, delivery or entry. Policy fits use only their four fitting seasons' labels. LOSO is a cross-season fitting-procedure screen, including later seasons in earlier folds, not an as-of historical deployment. Proper scores are unchanged by construction; forecast-byte parity is checked, and unchanged scores cannot reject this policy candidate.
- **Pairing:** within each (seed, season, date), use rank 1 and the first retained lower-ranked candidate with a valid different game identifier. Absent partners are explicit and never count as a two-hit success. Required identities, ranks, probabilities and labels are validated; malformed, duplicate or ambiguous rows are not silently skipped.
- **Classifier:** one five-bin cutpoint array per A1/A2 fold, computed from pooled rank-1 probabilities on its four fitting seasons. That array is frozen and used in both phases, all replays and all projections; equality with a cutpoint enters the upper bin. Early/late frequencies and hit rates are estimated through those same cutpoints, with late defined by the final 30 calendar days under §4. Both policies use identical rates and classifiers. A0 keeps its own saved classifiers. No held-out re-quantiling, phase-switching of A0 cutpoints or fallback bin scheme is allowed. Empty or degenerate fitting bins stop the screen before result inspection.
- **Rates:** single rates use available primary days; double-success rates use actual primary/partner joint outcomes on partner-eligible days. Partner availability frequencies are recorded separately and the execution clamp is applied in replay and projection. Any simplifying availability assumption made by the optimizing solvers is shared by A1/A2 and disclosed; their evaluated values use legal execution.
- **Historical split:** hold out each of 2021–2025 in turn, including all 24 seeds. Fit A1/A2 only on the other four, then freeze them before replaying the held-out season. A0 is unchanged. Its tail was fitted on all five seasons and its base's original fitting profiles are unavailable; comparisons against it retain that exposure limitation.
- **Population and final fit:** average seeds within each season, then weight the five seasons equally. Freeze the final decision-object rule now: the same five-bin, 30-day-phase recipe fitted once on all 2021–2025 profiles. Its artifact, manifest and separately labelled projections are produced before D7. Fold results evaluate the fitting procedure; they are not held-out measurements of that final all-season fit.
```

### E4 — replace §4 and §7

```markdown
## 4. Replay, calendar, parity and Δ stress
- **Base:** the July `replay_vectorized` semantics, with the corrections below. A scalar synthetic reference verifies the corrected replay independently.
- **Calendar:** freeze an outcome-free calendar per season, specifying contest opening date, inclusive final date, exclusive end date and evidenced no-opportunity dates. Raw days left is max(0, exclusive_end − current_date) in calendar days. A0 applies its saved horizon cap for both routing and lookup, without dropping later profile rows. A1/A2 share the actual registered season horizon. All arms consume calendar time on skips and absent rows; neither the clock nor the 30-day phase is inferred from profile row position or observed-date count.
- **Availability:** inspect the frozen ranked pool on every observed date, including dates an arm skips. Primary unavailable means no play; a requested partnerless double executes as a single on the primary outcome. A missing playable date is unknown availability, not an evidenced skip or no-game day. Counts and coverage follow §7. The replay remains conditional on the offline participant/pool information set.
- **State:** each arm receives its own running m, retains it across misses and saver use, and applies the cap. A0 includes its original tail routing, own classifier and trusted-best stop behavior. No fix is applied only to A2.
- **Parity before fixes:** reproduce the July Δ=0 deployed-base, always-single and old always-double anchors, including resets, at the original implementation's tolerances (`ANCHOR_DIFFGAME`, `ANCHOR_TOL`). This diagnostic alone retains the old 180-position clock and partnerless +2 behavior. Failure stops before any corrected outcome result is read.
- **Fix ladder:** on the fixed reference arms, report cumulative effects in the order calendar, legal demotion, m-state plumbing, A0 tail routing. The m-state plumbing step can be inert for m-independent arms; it is not empirical validation of m correctness. Candidate decisions use the final corrected evaluator only.
- **Stress:** Δ ∈ {0, 0.05, 0.10, 0.139}. The A1/A2 tables fitted at Δ=0 remain fixed in every stress; no stressed re-solves are credited to them. For each fold, define r_bar from fitting-season partner-eligible leg outcomes. Thin only eligible partner hits with probability min(1, Δ/r_bar), using 200 replicates and RNG seed 20261004. Primary outcomes are unchanged, including partnerless single fallbacks. This is a uniform marginal-leg stress with random realized reduction, not an exact conditional bin haircut or an estimate of live calibration. If r_bar is zero, the stress is unavailable and cannot pass its gate.
- **Randomness and precision:** use one outcome mask for every arm on a given seed/date/replicate; reuse uniform draws across Δ for nested thinning. Bind streams to stable seed/season identity, not filesystem enumeration. Report achieved marginal and primary-hit-conditional haircuts and Monte Carlo SE separately from season variation. Average thinning replicates inside each seed before season aggregation. A gate within two Monte Carlo SEs of its threshold is inconclusive; replicate count and seed are not changed after seeing the result.
```

```markdown
## 7. Family, stopping and missingness
- **Family:** one candidate, with two compulsory paired acceptance contrasts (objective and switch); both must pass. No selection among contrasts, folds, stresses or comparators. No formal significance or multiplicity-adjusted confidence claim is made; all secondary projections are descriptive.
- **Stopping:** one registered historical run; no retuning of bins, phase, final-fit rule, thresholds or stress parameters after results. Failed provenance, schema, calendar, parity or independent evaluator checks stop before the corresponding result is read. A software repair requires a recorded invalidation and reviewed correction; it does not authorize selecting among result runs. Historical disposition cannot bypass the applicable C1 2027 validation requirement, independent acceptance or D7.
- **Missingness:** require all five seasons and all 24 registered seeds. Do not silently drop a missing profile, seed, primary or ambiguous row. Distinguish observed availability, evidenced no opportunity and unknown coverage on the frozen calendar. Unknown playable-date coverage makes acceptance inconclusive; a conditional replay may be shown descriptively with those dates explicitly treated as no play, retaining calendar time. Report counts by season and seed, no-primary/no-partner dates, partner ranks, excluded rows, and availability on arm-skipped dates. No imputation or extra-calendar opportunities.
```

### E5 — replace §5 and §8

```markdown
## 5. Metrics and the fixed-policy projection
- **Paired streak-value contrasts:** A2 − A1 (objective) and A2 − A0 (switch), using identical held-out sequences. Average within season across seeds, then average the five seasons equally. Report all five season contrasts and their range. Any SE across the five season means is labelled descriptive, conditional on overlapping fitted folds; it is not a validated sampling CI for 2027. An across-profile SE may appear only as conditional seed-variation diagnostics, never as the main table's uncertainty or a power claim.
- **Secondary for every arm and every Δ:** capped E[season best], reach-20/30/40 and observed reach-57 counts, resets, play/single/double/skip counts, legal demotions and the A1 − A0 adaptation contrast. Reach probabilities in replay mean historical trajectory proportions, conditional on the frozen participant surface.
- **P(57):** exact fixed-policy evaluation conditional on a declared transition model, retaining s, m, days, one-use saver and the environment's quality/availability information. Fixed means lookup the registered policy; never optimize again during evaluation. A0's tail routing and all legal clamps are included. Each cell is a model projection, not a measured jackpot frequency.
- **Comparable environment:** for each fold, estimate rates and frequencies twice: from fitting seasons and from its held-out season. Classify both through frozen cutpoints. Use a common refinement of A0's saved bins, A1/A2's bins and primary/partner availability, so each arm receives the same environment and executes through its own classifier. No policy's original or embedded training rates stand in for another arm's evaluation environment. Empty unsupported rate cells make the relevant projection unavailable, not zero or imputed. Report paired A2−A0 and A2−A1 jackpot differences in probability and percentage-point units; relative changes are unavailable when the reference is zero.
- **Calibration grid:** c ∈ {0, 0.02, 0.04}, subtracting c from both primary and joint-success rates with a zero floor. Enforce 0 ≤ p_both ≤ p_hit ≤ 1 after every transform. Apply identical transformations to every arm without refitting actions.
- **Dependence grid (speculative stress, not a fitted correction):** alongside iid, carry an exogenous run counter r, initially zero, capped at 8, for consecutive calendar-day primary hits. It is updated from the primary outcome even when an arm skips. A no-primary/no-opportunity day resets it. For h ∈ {0, 0.02, 0.04}, when r=8 reduce the already calibration-stressed primary rate by h with a zero floor, and multiply the joint-success rate by the new-primary/old-primary ratio (zero when the old primary rate is zero). Draw the coherent outcome categories primary miss, primary-only hit and joint hit; update r on every such draw. This shared outcome-history environment creates serial dependence without making the environment depend on an arm's played streak. It also changes unconditional calibration and is explicitly not a marginal-preserving estimate of the July run suppression. h=0 reproduces iid. Evaluate the full c×h grid at Δ=0 and Δ=0.10, with the analytic conditional-leg stress p_both := max(0, p_both − Δ·p_hit) before the dependence transform; label this separately from replay's marginal thinning.
- **Validation:** A1 equality with `solve_mdp` is required on synthetic matching rates, horizons, phase frequencies and an all-partners-available kernel; its matched tail changes no unreachable-region jackpot value. It is only a check of the m-independent slice. Independent tiny-path enumeration must also verify an m-dependent policy, retaining m on a reset, saver catch and consumption, no repeated saver, skip-time advancement, early/late transition, bin-boundary equality, partnerless +1, and a double crossing the target. Test A0's equality boundary s+2d=57, both sides of tail routing, and its trusted-best stop. Test the new phase-aware whole-season E[best] solver against independently enumerated tiny objectives; the current stationary tail solver and its loader equality are not sufficient oracles for that extension. Failed checks stop before outcome results.
- **Decision consequences:** report executed-action differences on common states twice: weighted by A0's visited (s,m,saver,date) states, and weighted by A2's visited states. Identify the denominator and report s≥10 separately under each visitation distribution. Also report own-trajectory same-date action divergence, labelled as including state divergence. These frequencies are descriptive, not rejection thresholds.
```

```markdown
## 8. The trade tables for Eric
Rows are A0, A2, A1, always-single and legal always-double. The historical table reports capped mean season best at Δ=0 and 0.10, paired objective/switch contrasts, all five season results and their range, reach-20/30/40, resets and action counts. Its uncertainty is season variation under §5, not an independent-n=120 SE.

A separate projection table reports each fold's fitting-rate and held-out-rate P(57) values and paired jackpot differences on the full registered c×h×Δ grid, with equal-season summaries and scenario ranges. Keep fitting-rate and held-out-rate summaries separate. Report the final all-five-season decision object's projections separately, identifying its hashes and disclosing their outcome exposure; these are not held-out estimates of that artifact. Unavailable cells remain unavailable. Do not select a favorable scenario for the headline or describe an unresolved cost as zero.

The sentence above the tables is: “The historical fitting-procedure screen changes mean season best by Z streaks versus A0 [five-season range], conditional on the retained participant surface and A0's prior exposure. Its jackpot-cost projections span [registered scenario results]; the true jackpot cost is unresolved. The separately identified final artifact has its own projection table. These tables do not alone waive C1's 2027 validation requirement or authorize a change to picks.”

Own expected season best remains the named objective. Reach-20/40 are descriptive thresholds, not estimates of prize-winning probability, official field eligibility or prize share.
```

### E6 — replace §6

```markdown
## 6. Thresholds and dispositions, fixed before outcomes
**Positive historical screen** requires all of the following on valid, complete inputs and the final corrected evaluator:
1. At Δ=0, equal-season mean A2 − A1 and A2 − A0 are each ≥ +0.25 streaks.
2. Each contrast is strictly positive in at least 4 of 5 seasons.
3. At Δ=0.10, each equal-season mean contrast remains ≥ 0 under the fixed policies.
4. At both Δ=0 and 0.10, pooled reach-20 for A2 is no more than 2 percentage points below either A1 or A0.
5. At Δ=0, neither streak-value contrast is below −0.25 in any season.
6. Required validation/provenance checks pass; no numerical gate is within the Monte Carlo ambiguity band in §4. The required projection tables are present, explicitly qualified and bound to the final decision object; unavailable projections cannot be presented as a quantified jackpot cost.

**Negative:** on valid complete inputs, either mean streak contrast is ≤0 at Δ=0, or a registered reach-20/individual-season regression guardrail is breached. **Inconclusive:** all other cases, including insufficient coverage, ambiguous stress precision or unmet validation requirements. No disposition selects another candidate or permits retuning.

**Rationale (reasoning only):** +0.25 is a pragmatic integration-benefit bar, not a measured MDE. Four strictly positive seasons tests directional spread, not statistical significance. Δ=0.10 is a declared leg-risk scenario, not a live calibration estimate. The 2 pp pooled milestone tolerance and −0.25 season limit are engineering regression limits; the latter follows the closest guardrail precedent. Five shared-date seasons and overlapping fits do not support an independent-120-trajectory power claim. Unchanged proper scores, low action-change frequency and iid P(57) gains neither reject nor accept the candidate.

Whatever the historical disposition, Eric gets the complete tables. Positive means the candidate survives the historical screen; independent acceptance, the applicable C1 2027 validation requirement and approval of the concrete policy under D7 remain outstanding. Historical positivity is not a recommendation to activate immediately.
```

### E7 — replace §9, the cost bullet, and the limits' effective-sample/optimism bullets

```markdown
## 9. Concrete decision object and later integration (D7)
- Before asking Eric to approve a policy change, materialize the frozen all-five-season fit as a reviewable decision object, with the full rates/classifier/calendar/recipe manifest and action-table SHA-256, and produce its projection table. This is separate from the historical fold results and its projections are not held out.
- A new sha-bound artifact (working name `mdp_season_best_policy.npz`) packages exactly that approved decision object. The loader validates the full-season phase/rates/classifier contract and independently re-solves for equality. Packaging may not change the fit or actions after approval; a changed decision object requires a new concrete review and D7 decision.
- Integration remains a separate tested, reviewed engineering step. Touch points include the tail-only horizon/loader assumptions, phase-aware whole-season solver, objective routing, season-end configuration, scheduler skip census, MDP alignment health, skip-policy shadow and boundary census. Checklist A2 must cover the new artifact before activation.
- Neither this registration nor its historical screen approves deployment, activation, a paid run, a commit or a push. Independent acceptance and the applicable prospective C1 gate remain required unless explicitly amended by Eric.
```

Replace the §10 cost bullet with:

```markdown
- **Compute sizing (reasoning, not measured):** five arms × five held-out seasons × 24 seeds × four Δ cells, with the registered thinning replicates, matched phase-aware solves and fixed-policy calibration/dependence projections. The proposed whole-season implementation has no measured “seconds” runtime. It must fit the declared 4 CPU-hour/3 wall-hour launcher limits and C1 caps; overruns follow the approved stop rules, without an automatic rerun or scope expansion.
```

Replace the effective-sample and optimism bullets in “Limits stated up front” with:

```markdown
- **Effective sample and uncertainty:** five historical seasons, re-used across 24 correlated seed profiles; overlapping fitted folds and prior exposure prevent treating either 120 profiles or five fold means as a validated 2027 sampling experiment.
- **Exposure and final fit:** A1/A2 exclude their evaluation season from new policy fitting; A0's tail used all five seasons and its base fitting corpus is unavailable. The historical data were previously examined. The final all-season artifact is a separate exposed fit; cross-validated procedure results do not become held-out measurements of that artifact.
```

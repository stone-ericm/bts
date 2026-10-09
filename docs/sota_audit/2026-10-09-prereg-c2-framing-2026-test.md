# C2 side item (e): catcher-grouped framing, 2026 out-of-sample test — pre-registration (revision 1, under design review)

**Status: under design review.** The BTS lead (bts-lead2) wrote it on 2026-10-09 at the herdr manager's request. Nothing here runs, and no 2026 data was read to write it. It is frozen at its reviewed commit before any code is written or any 2026 file is read. The review is a fresh Codex session with memories off, at most two rounds.

**Authority:**
- **Recorded:** Eric chose this test on 2026-10-09 at 16:05 EDT. He typed "Option 1" in the lead's pane (register row C2-framing-2026-test).
  - Option 1, as posed, carried an exception for this one contrast to three records:
    - D3 = RESERVE: "every new idea from this wrap validates prospectively in 2027" (plan, 9/14, "reversible by Eric").
    - The side-item ruling's scope: "worth testing on untouched 2027 data", never "ship it" (register row C2-framing-side-item, Eric 10/06).
    - D2 as ruled ("as recommended"), whose recommendation asks for "earlier-fit/later-untouched 2027 validation" before a forecast change affects picks.
  - The register records Eric's complete words as "Option 1". It separately records the lead's interpretation that selecting the option selects its stated exceptions; Eric did not separately type those exceptions. Under that recorded interpretation, 2026 stands in as the later, contrast-unread season for this contrast. The other D2 gates and D7 remain in force.
  - Option 1 as posed also left B out.
- **Still needed before the first launch:**
  - A compute allowance in Eric's own typed words, verified by the manager (§7). The shared C1 + C2 ledger has 27.31 CPU-h left under its 165 cap, which is not enough.
  - The reviewed code.
  - An exposure row, X-37 PREDECLARED, naming every 2026 input, recorded before any 2026 read.

## 1. The question
Does variant A, the screen's catcher-grouped framing measure replacing production's `pitcher_catcher_framing`, beat the screen-recipe baseline on 2026 when its prediction-day catcher is the final-boxscore starter proxy specified below? A-posted tests that proxy, and A-projected tests the fixed historical projection. Neither reconstructs which opposing catcher production actually knew at its pick time. A positive is conditional evidence for the candidate; operational availability remains a separate gate before adoption.

**Why 2026.** 2026 has been read for other questions (register rows X-01 to X-30, including W1.2's walk-forward of the production recipe). It has never been read for any catcher-grouped framing measure: the screen read 2017–2025 only (X-35, X-36). Variant A was chosen on 2024 and 2025 alone.

## 2. Arms (per seed)
1. **Baseline:** the production feature set, as in the screen.
2. **A-posted (primary; this arm decides):** A, with each predicted day's catcher taken from the final-boxscore starter proxy (§4).
3. **A-projected (stress arm; reported, does not decide A):** A, with each predicted day's catcher taken from the earlier-date projection (§4).

**B is not carried.**
- It was inconclusive in the screen (4 of 10 seeds).
- It would add about a third more compute.
- Running a second candidate "for information" on the one untouched season invites choosing between A and B after the results are seen.

## 3. Inputs (complete read inventory, all pinned by sha256 at admission)
- **The PA parquets** `pa_2017`–`pa_2025`: the screen's pins, unchanged.
- **`pa_2026`:** a frozen copy of the box's current build. Regular season only.
- **The probable-pitcher lookup,** frozen over the pinned 2017–2026 games. Its historical part retains the screen's entries. Its 2026 entries contain the pitcher ids from `gameData.probablePitchers` and the away/home team ids from `gameData.teams`; `compute_all_features` needs both to build `opp_bullpen_hr_30g`. Missing coverage is counted, with no live scan or cache fallback.
- **The raw-feed sources** for the 2026 lookup and the new 2025–2026 catcher table are acquisition inputs too. Before any 2026 source is opened or hashed, X-37 names the source paths and permitted extraction. Admission then pins the derived files and a source manifest listing each raw path, sha256, game id and season, plus counts and a digest. Hashing and JSON decoding read whole source files; only identity/schedule fields are extracted or inspected, with no play results, batting/pitching statistics or other outcome analysis. The 2025 sources supply projection history only.
- **A new starter-proxy table** for regular-season 2025–2026 games whose game ids occur in the pinned PA parquets for those seasons. Read only those source feeds; do not include canceled/unplayed raw-feed records or scan unrelated seasons. It has exactly one record per `(game_pk, fielding_side)`, including unidentified sides. It records `gameData.game.pk`, season and type, `gameData.datetime.officialDate`, team id, game number, catcher id or null, and identification reason. Duplicate/conflicting game-side records refuse admission.
  - A candidate must have `battingOrder` exactly one of the strings `100`, `200`, …, `900`, a positive integer player id (no string/float/bool coercion), and a nonempty `allPositions` array whose first entry has `code == "2"`. Use the array's given order. Do not use the player's primary `position`, an arbitrary first player, or any later C entry as a fallback. Exactly one qualifying candidate identifies the proxy; zero or multiple candidates give null and a reported reason.
  - History is ordered by `(officialDate, game_number, game_pk)`. Use a positive integer `gameData.game.gameNumber`; when absent use 1 and record that fallback. Malformed dates, team ids, game ids or supplied game numbers refuse admission. A raw feed's `allPositions` order is a proxy assumption, not independent proof of its first defensive alignment. A malformed player record or position array makes that side unidentified with a recorded reason; it must not be silently ignored to manufacture a unique candidate.
- **Read closure.** The runtime replaces the live probable-pitcher scan and park-drag attach before feature computation, as in `screen.install_closed_inputs`. Model caching and per-PA output accumulation are disabled. Retained profiles are read back for scoring/reconciliation; code, admission, review, exposure and C1 operational records are process inputs, not additional baseball data. No other baseball input, live lookup, raw-feed fallback or external park table may be read. The ten seed values in §5 are constants, so no seed-set file is read.

## 4. The catcher, by arm, and the missing-catcher rule
**Training rows** (every season, and the 2026 days before the predicted day) use the catcher-grouped feature exactly as screened:
- It is keyed by the build's `fielding_catcher_id`.
- As in `screen.framing_by`, first take the mean of `pa_borderline_csr` over all PA rows for each `(fielding_catcher_id, date)`, ignoring missing PA rates. Both games of a doubleheader share one date aggregate. Then sort dates and apply `shift(1).expanding(min_periods=5).mean()` per build catcher id. Catcher history starts in 2019. The minimum is five nonmissing daily rates, not merely five dates or five games. A date whose daily rate is NaN does not count toward that minimum.

**Predicted-day rows** carry the same feature, looked up for the arm's named catcher as of the predicted day:
- **A-posted:** the table's final-boxscore starter proxy for the opposing fielding side of that game: away for `is_home == True` batter rows, home otherwise.
- **A-projected:** for that opposing team id, select its last 10 table games with `officialDate < predicted_date`, across home and away games and across 2025–2026, in the order defined in §3. Use all available games if fewer than 10 exist. Take this game window first, then ignore unidentified catcher ids within it; do not extend the window to obtain 10 identified catchers. Same-day games never enter history, including game 1 when predicting game 2.
  - Choose the most frequent identified catcher id in this window. Resolve a frequency tie by the tied catcher whose latest start has the greatest history-order tuple. If no identified catcher remains, the projection is missing.
  - The home-game proxy comparison below motivates this fixed heuristic. It does not validate its accuracy against true starters or against production's previous-game rule over all team games. No rule is retuned on 2026. No roster or transaction filter is applied; a prior-year catcher may have left the team, so an identified projection is not proof of roster eligibility.
- **The value looked up:** form the same build-keyed daily rates as the training path, retaining only dates strictly before the predicted date. Apply `expanding(min_periods=5).mean()` to that ordered prefix and take its last value, or NaN for an empty prefix. Do not take the last already-shifted feature row, which omits the catcher's last observed date. No same-date aggregate enters either game of a doubleheader. History ids remain the build's ids, never the arm's table ids. For a catcher id present in the build's group on the predicted date, this equals `framing_by` at that key/date; actual catchers omitted by the build have no such equality guarantee. It is also defined for an absent named catcher. The cutoff is by official date, subject to the suspended-game availability limit in §10.

**The missing-catcher rule** (both arms): the feature is missing (NaN) when any of these hold. LightGBM handles a missing value as production handles an absent lookup key. No substitute value is used.
- no starter proxy can be identified for the opposing side-game (A-posted);
- no identified catcher remains in the prior-game window (A-projected);
- the named build catcher id has fewer than five nonmissing daily rates strictly before the predicted date.

Report these mutually exclusive reasons in the order above, per distinct opposing side-game and per predicted PA row, with denominators; also report identified ids and finite-value coverage. Missing cases never retain the build's prediction-day feature by accident.

**Why the arms differ only on predicted-day rows.** The shared training frame retains the screened finished-game proxy identities. On each retrain the walk-forward adds earlier official dates from 2026, subject to §10's event-availability limit. The arm's catcher replaces the build's catcher only on the day being predicted. This needs a hook in `blend_walk_forward`:
- **Where it applies:** after selecting/copying `day_data` and fitting from the untouched `available` frame, but before the first blend prediction. The modified copy supplies both the blend predictions and `_estimated_pa_game_predictions`, including its representative starter rows and copied reliever rows. Replace only `catcher_framing`, for every prediction-day PA; keep row count, index, order, ids, dates, other features and labels identical. Never write arm values into `df`, `test_data`, `train_pool` or `model_train_window`, or recompute framing history after changing catcher identities.
- **Off by default,** and tests must show existing callers are bit-identical; training inputs are equal between the A arms and unchanged on later retrain dates; both prediction paths see the override; missing lookups become NaN; and observed-key as-of values equal `framing_by`, including missing daily rates and doubleheaders. Prediction exceptions or missing model scores refuse acceptance rather than silently averaging fewer blend models.
- **Reviewed as a `src/` change.** It is research plumbing that changes no production behavior, but it is a change under `src/`.

**Measured on 2024–2025** (the lead, 2026-10-09, reproduced in design review): venue-based home-game sequences of the build's catcher, retaining venues with at least 30 games and scoring after 10 home games of history. These are observed proxy ids, not verified starter identities.
- **Previous-home-game guess:** right in 968/2,117 = 45.7% of 2024 scored home games and 883/2,121 = 41.6% of 2025's.
- **Most-used-in-last-10-home-games guess:** right in 1,223/2,117 = 57.8% (2024) and 1,156/2,121 = 54.5% (2025). The original comparator includes same-date history for 23 and 26 predictions respectively.
- **The top build catcher's share in these venue sequences:** median 62% (2024) and 58% (2025), over 30 venues per season.
- **The build's catcher** appears among his side's first nine observed PA batter entries in 4,556/4,850 = 93.9% of 2024 team-games and 4,580/4,850 = 94.4% of 2025's. This is not a starting-lineup reconstruction: early substitutions can change those entries. The other 294 and 270 cases demonstrate a proxy disagreement, not an exact measured count of nonstarters.
- Consequently this comparison supports a provisional heuristic choice only; it does not measure the registered all-team-game, earlier-date projection. Its rule stays fixed through the 2026 test.

## 5. Procedure
**The run:**
- Test season 2026 only.
- Blend walk-forward as in the screen: the 12-model blend, retraining every 7 test dates, `estimated_pa` basis, `BTS_LGBM_DETERMINISTIC=1`, the seed as `BTS_LGBM_RANDOM_STATE`, and the screen's feature settings, re-checked against production's at admission.
- Labels rebuilt with the resumed portion excluded (BTS scoring), as in `screen.original_portion_labels` and `screen.relabel`; void batter-games are dropped and ranks renumbered. Require a boolean `is_resumed_portion` column with no missing values in `pa_2026`, and record its presence and flagged/unflagged row counts. An absent or malformed flag refuses admission rather than certifying exclusion from an empty count.
- The screen's self-check against `pitcher_catcher_framing`, feature-settings checks, input closure, top-10 profiles, scoring settings (10,000 simulation trials and 180 simulated days), and retained-profile reconciliation remain required. A changed production feature setting at admission stops for a design decision; it is not silently substituted.

**Seeds:** the screen's ten, positions 0–9 of `canonical-n10.json`, with one run per seed, in this literal order: 2273360, 260991262, 1746737973, 2048, 3629294338, 1277948386, 3219332220, 2207587974, 3170105529, 2675988121. No other seed or replacement run enters the aggregate.
- They run in order, one at a time, through the C1 launcher. Seed 1 runs first and is checked for cost.
- No launch from 00:45 to 03:10 box clock, and the manager gets a heads-up before each launch.

**Sealing:**
- Outcomes stay sealed until all ten seeds have reconciled.
- Opening them is announced to the manager first, with the ledger total and every TERMINAL/RECONCILED record, as in stage two.
- There is no early stop on outcomes, only on cost (§7).

## 6. Dispositions (fixed now): A-posted against the baseline, all ten seeds
**Completeness first:** derive and retain the expected evaluation calendar from the pinned 2026 regular-season PA rows after original-portion filtering, before arm scoring: the sorted official dates with at least one scoreable batter-game. Every arm and seed must retain exactly this nonempty date set, with exactly one valid rank-1 row per date after relabeling. Validate binary hits, finite probabilities, ranks, identities and the registered settings, and reconcile metrics to retained profiles. A shared omission is incomplete too. If A-posted has no finite catcher value on any predicted PA in the complete season, the catcher contrast is unavailable and incomplete, even if predictions from missing-value routes can be scored. Never inner-join unequal calendars, average available seeds, or accept swallowed prediction failures.

**Quantities:**
- For seed s and date t, x(s,t) is A-posted's rank-1 hit minus baseline's rank-1 hit. d(s) is its mean over the complete calendar; m is the mean of the ten d. All calculations use hit fractions without rounding; +0.3pp means 0.003.
- L = the 10th percentile of m under paired day resampling. With dates ascending and seeds in §5's order, generate one `(10000, N)` array using `default_rng(20261009).integers(0, N, size=(10000, N))`, where N is the calendar size. Each row draws N days with replacement. Apply its indices jointly to all ten seeds and recompute the mean of their deltas; take `numpy.quantile(draw_means, 0.1, method="linear")`. Do not retrain models, re-rank candidates or bootstrap seeds.
- Compute and retain the same descriptive m, L and positive-seed count for A-projected against baseline on that calendar, without another adoption decision. Report discordant/positive/negative/zero daily mean deltas, ties and any constant bootstrap distribution.

**Dispositions:**
- **positive** requires all three:
  - m ≥ +0.3pp;
  - L > 0;
  - d > 0 on at least 6 of 10 seeds.
- **negative:** m ≤ 0.
- **inconclusive:** anything else.
- **incomplete:** any failed completeness or validation condition, or anything other than exactly the ten registered seeds, each validated. This takes precedence over every numerical disposition. Zero seed deltas do not count as positive; L = 0 fails the positive rule. Complete all-zero deltas are negative by the fixed m ≤ 0 convention, which includes a tie and does not imply measured harm.

**Why L.** The paired day is the sampling unit for a conditional one-season contrast; seeds are not independent season replications. On 2024 and 2025, A and baseline chose the same top batter-game on 80.9% and 74.5% of seed-days. The analytic iid-day standard error of the ten-seed daily mean, sample SD divided by √N, is 1.03pp (185 dates) and 1.43pp (184 dates); the mean single-seed standard errors are 1.91pp and 2.42pp. Averaging seeds reduces some noise but leaves shared day noise. This does not establish that day noise is the largest source of uncertainty or that dates are independent.

**Dependence sensitivity, descriptive only:** also report a seven-test-date circular moving-block resampling L, preserving each day's ten-seed vector. Reset `default_rng(20261009)`; draw a `(10000, ceil(N/7))` array of starts with `integers(0, N, ...)`, append seven successive indices modulo N for each start, and truncate each draw to N indices. Use the same `method="linear"` quantile. Its disagreement with the iid result is reported prominently and cannot be treated as an independent confirming test. Neither resampling accounts for between-season change, seed-source selection, feature choice or lineup-proxy error.

**Illustrative resolution** (reasoning, not measured 2026 power): treating the iid L criterion alone as a one-sided normal cutoff gives `Phi(effect / SE - 1.28155)`. At an assumed SE of 1.23pp, midway between the two observed season SEs, this is about 62% at +1.95pp, 32% at +1.0pp and 10% at zero. Across SEs of 1.03–1.43pp the first two ranges are about 53–73% and 28–38%. These are not power estimates for the complete three-part disposition: the practical-effect and seed-vote requirements can reduce them, and sparse discordances, dependence or degenerate draws invalidate the normal approximation. The screen-selected +1.95pp is an optimistic scenario, not an unbiased expected effect. A positive is a screening disposition, not a guaranteed 10% false-positive test.

**A-projected** has no weight in A's disposition and does not approve a production fallback. When A-posted is positive, it informs a proposed fallback for later gating:
- If its m against baseline is > 0, the projection remains a candidate for the production fallback; report its m, L, seed signs and missing coverage before any usability claim.
- If its m is ≤ 0 (including a tie), the note recommends investigating a posted-only catcher value, with missing values before posting.
- In either case, the actual mixed posted/projected or posted/missing serving rule needs its own operational parity and practical-effect/regression stress checks and independent acceptance before D7. A nonnegative point estimate proves neither noninferiority nor usability; the missing-value fallback is not tested merely by scoring games with naturally unidentified catchers.

**Streak metrics** (`mean_max_streak`, exact P(57) and the replay longest streak) are reported per seed for both A arms and carry no weight in the disposition. The note states first whether they moved against P@1, for two reasons:
- D1's objective is the longest streak.
- The screen's streak simulation fell on 8 of 10 seeds under A.

## 7. Compute, budgets and stops
**Estimate** (reasoning, not measured on 2026): about 2.75 CPU-h per 2026 walk-forward.
- The supplied 2025 walk-forwards averaged 2.3799 CPU-h (30 units); 2024 averaged 2.0140 after excluding the single seed-1 A-2024 unit at 7.0429 CPU-h, described in the accepted note as a production overlap (29 units). Including it, 2024 averaged 2.1816. Extrapolating the filtered year-to-year increase once gives 2.7457, rounded to 2.75; this is a planning assumption, not a 2026 measurement. Feature building, scoring, table preparation, reconciliation and launcher overhead also count against the allowance.
- That gives about 8.3 CPU-h per seed for three arms, and about 83 for ten seeds.
- For comparison: baseline and A-posted alone are about 55; adding B would make it about 110.

**Proposed budgets** (Eric's to set, in his own typed words, before the first launch):
- 12 CPU-h per seed.
- A first-walk-forward stop on each seed after its baseline-2026 unit completes above 4.1 CPU-h (the rounded proposal, approximately 1.5 × the estimate). The C1 guard separately enforces the per-seed hard CPU budget.
- A total additional allowance of 100 CPU-h, with the effective prior shared total (including off-launcher aggregates) carried into the manager-verified cumulative cap, or an explicitly authorized separately enforced allowance. The existing 165 cap does not permit this plan by itself. The launcher must reserve the full per-seed declared budget against the authorized remaining total before each launch.
- Any cost stop, killed/incomplete seed, or exhaustion of that allowance stops and reports; no replacement seed, rerun or further launch without Eric's decision. Costs may be read while outcomes remain sealed.

The box is fixed-price, so the cost is $0 of new spend.

## 8. Process and reporting (as stages one and two)
1. This design is reviewed by a fresh Codex session (memories off) and frozen at its reviewed commit.
2. The code is written test-first: the catcher-table builder, the walk-forward hook, and the 2026 admission and aggregate. It gets a Codex code review (memories off) and is frozen.
3. X-37 is recorded before any 2026 file is opened or hashed. Then the declared 2026 inputs, lookup and catcher table are prepared and pinned in the admission, with the raw-source manifest; no unregistered source is read.
4. Seed 1 runs, then its cost is checked, then seeds 2–10 run.
5. The aggregate runs, then the results note is written.
6. An independent acceptance follows: a fresh Codex session, memories off, Part 1 blind.
7. The result goes to Eric in the lead's pane, plain language first, then to the manager.

## 9. What a result can support
**A positive** is one season of candidate-specific evidence conditional on the registered offline recipe and final-starter proxy. The recorded interpretation of Eric's option changes the validation-season requirement for this contrast; it does not waive D2's practical-effect/regression limits, applicable stress tests or independent acceptance, and does not approve a fallback, deploy or activation. Those remaining gates must pass before a concrete change is put to Eric under D7. The production candidate has two parts:
- a catcher-id lookup implementing the registered prior-date rate, with its serving freshness and event availability checked;
- the opposing catcher on each slate slot from a pre-pick posted lineup, with a separately gated projection or missing-value rule informed by §6.

This test does not specify or pass the actual mixed serving rule's regression limits, assess the deployed selection policy's downstream value, or prove that the final-starter proxy was available at pick time. A P@1-positive result alone settles none of those gates.

**Inconclusive or negative:** the baseline stays. A 2027 shadow remains possible.

## 10. Limits
- **One season.** Day resampling describes 2026's day-to-day noise. It cannot show the effect carries over to 2027.
- **The posted catcher is a final-boxscore starter proxy.** A late scratch may replace the catcher that was posted or known at pick time; the final feed can retain a different identity. The table cannot reconstruct the posting time, identity revisions or scratches. The inspected lineup collector records confirmation times, not player ids; the prediction and commit path has no opposing-catcher field. No archived pre-pick opposing-catcher source is an input here.
- **Production picks are mixed.** The normal lock gate tracks the selected primary and double-down batters' projected-lineup status, not confirmation of their opponents' catchers. Other candidate games can remain projected. The two A arms are separate stress endpoints; neither reproduces or mathematically bounds the rank-1 performance of a mixed slate. A hybrid can choose a different batter and perform worse or better than both.
- **Only the catcher is changed.** Everything else about the walk-forward is as in the screen: it still uses each day's realized PA rows (actual pitchers, actual lineups).
- **Serving freshness differs.** `_build_feature_lookups` takes the last nonmissing already-shifted value for historical features such as pitcher framing, omitting the last contributing entity date and possibly more dates when values are missing. The proposed as-of catcher lookup includes that last prior daily rate. Static, current-slot and differently computed features do not all have a one-appearance lag; serving parity still needs its own check.
- **Historical catcher attribution stays a proxy.** The build selects the first player with any C entry in `allPositions`, with a primary-position fallback, and assigns that id to every opposing PA in the game. It can pick a substitute and does not follow in-game catcher changes. The arm corrects prediction-day identity only; its historical rates retain the screened attribution.
- **Suspended-game event availability remains bounded.** Features and training retain resumed-portion PAs at the original official date, as in the screen. Even strict earlier-date cutoffs can admit those events before they occurred. Rebuilding original-portion evaluation labels does not repair feature-history availability. This test establishes no unconditional pre-pick availability.
- **Streak and policy limits.** The screen scorecard simulates the `combined` strategy for 180 resampled days and computes an exact P(57) under its quality-bin/Markov assumptions; its chronological replay is separate. These are not validated reach probabilities for the deployed policy or D1's 2027 objective. Streak regressions are reported and carry no weight in this disposition; any downstream trade and production regression limits remain part of the concrete D7 proposal.
- **The seeds** are the screen's ten and keep its limits (outcome-ranked source set; 32-bit wraparound of five seeds).
- **2026 is not untouched in general.** W1.2 read the production recipe's 2026 walk-forward level. The A-versus-baseline contrast is what remains unread.

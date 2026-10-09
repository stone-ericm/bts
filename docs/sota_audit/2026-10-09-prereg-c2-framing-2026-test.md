# C2 side item (e): catcher-grouped framing, 2026 out-of-sample test — pre-registration (revision 1, under design review)

**Status: under design review.** The BTS lead (bts-lead2) wrote it on 2026-10-09 at the herdr manager's request. Nothing here runs, and no 2026 data was read to write it. It is frozen at its reviewed commit before any code is written or any 2026 file is read. The review is a fresh Codex session with memories off, at most two rounds.

**Authority:**
- **Recorded:** Eric chose this test on 2026-10-09 at 16:05 EDT. He typed "Option 1" in the lead's pane (register row C2-framing-2026-test).
  - Option 1, as posed, carried an exception for this one contrast to three records:
    - D3 = RESERVE: "every new idea from this wrap validates prospectively in 2027" (plan, 9/14, "reversible by Eric").
    - The side-item ruling's scope: "worth testing on untouched 2027 data", never "ship it" (register row C2-framing-side-item, Eric 10/06).
    - D2 as ruled ("as recommended"), whose recommendation asks for "earlier-fit/later-untouched 2027 validation" before a forecast change affects picks.
  - Under this design, 2026 stands in as the later, untouched season for this contrast.
  - Option 1 as posed also left B out.
- **Still needed before the first launch:**
  - A compute allowance in Eric's own typed words, verified by the manager (§7). The shared C1 + C2 ledger has 27.31 CPU-h left under its 165 cap, which is not enough.
  - The reviewed code.
  - An exposure row, X-37 PREDECLARED, naming every 2026 input, recorded before any 2026 read.

## 1. The question
Does variant A, the screen's catcher-grouped framing measure replacing production's `pitcher_catcher_framing`, beat the production baseline on 2026? The catcher is the one production would actually know before the pick.

**Why 2026.** 2026 has been read for other questions (register rows X-01 to X-30, including W1.2's walk-forward of the production recipe). It has never been read for any catcher-grouped framing measure: the screen read 2017–2025 only (X-35, X-36). Variant A was chosen on 2024 and 2025 alone.

## 2. Arms (per seed)
1. **Baseline:** the production feature set, as in the screen.
2. **A-posted (primary; this arm decides):** A, with each predicted day's catcher taken from the posted lineup (§4).
3. **A-projected (stress arm; reported, does not decide A):** A, with each predicted day's catcher taken from a pregame projection (§4).

**B is not carried.**
- It was inconclusive in the screen (4 of 10 seeds).
- It would add about a third more compute.
- Running a second candidate "for information" on the one untouched season invites choosing between A and B after the results are seen.

## 3. Inputs (complete read inventory, all pinned by sha256 at admission)
- **The PA parquets** `pa_2017`–`pa_2025`: the screen's pins, unchanged.
- **`pa_2026`:** a frozen copy of the box's current build. Regular season only.
- **The probable-pitcher lookup,** frozen over 2017–2026 games. The 2026 part is built from `gameData.probablePitchers` only.
- **A new starting-catcher table** for 2025–2026, built once on the box from the raw feeds. The builder hashes every feed it reads and records the count and a digest. Per game and side, the table records:
  - the team id and the game's date and order;
  - the starting catcher: the starting-nine player (battingOrder ending `00`) whose first listed game position is C.
  - The builder reads identity fields only (teams, battingOrder, positions) and no outcome field.
  - 2025 is included only so the projection (§4) has history at the start of 2026.
- **Nothing else.** Production's live lookup scan and the park-drag table are replaced, as in the screen.

## 4. The catcher, by arm, and the missing-catcher rule
**Training rows** (every season, and the 2026 days before the predicted day) use the catcher-grouped feature exactly as screened:
- It is keyed by the build's `fielding_catcher_id`.
- Each value is the expanding mean of the catcher's per-date borderline called-strike rates over his prior game dates (catcher history starts in 2019), with a minimum of 5 dates.

**Predicted-day rows** carry the same feature, looked up for the arm's named catcher as of the predicted day:
- **A-posted:** the posted-lineup catcher, taken as the table's starting catcher.
- **A-projected:** the opposing team's most-used starting catcher in its previous 10 regular-season games on earlier dates.
  - Games whose starting catcher the table could not identify are skipped.
  - At the start of 2026 the window reaches back into 2025, and a tie goes to the more recent catcher.
  - This rule was chosen over production's existing projected-lineup rule (the previous game's lineup) because it guessed better on 2024–2025 (measured below).
- **The value looked up** is the named catcher's catcher-grouped framing as of the predicted day: the expanding mean of his per-date rates over all his game dates before that day, with a minimum of 5. For a catcher who caught that day, it equals the training-path value. It is also defined for a named catcher who did not catch that day.

**The missing-catcher rule** (both arms): the feature is missing (NaN) when any of these hold. LightGBM handles a missing value as production handles an absent lookup key. No substitute value is used.
- no starting catcher can be identified for the side-game (A-posted);
- no prior game exists (A-projected);
- the named catcher has fewer than 5 prior game dates.

Every case is counted per arm and reported.

**Why the arms differ only on predicted-day rows.** Production trains on finished games, whose catcher is known. The walk-forward feeds each finished 2026 day back into training. So the arm's catcher replaces the build's catcher only on the day being predicted. This needs a hook in `blend_walk_forward`:
- **Where it applies:** to the copy of the predicted day's rows, before the blend predicts and before the estimated-PA step builds its starter and reliever rows from them. It changes only the catcher-grouped column. Labels and training rows are untouched.
- **Off by default,** and a test must show existing callers are bit-identical.
- **Reviewed as a `src/` change.** It is research plumbing that changes no production behavior, but it is a change under `src/`.

**Measured on 2024–2025** (the lead, 2026-10-09; home-game sequences of the build's catcher):
- **Previous-game guess:** right in 45.7% of 2024 games and 41.6% of 2025 games.
- **Most-used-in-last-10 guess:** right in 57.8% (2024) and 54.5% (2025).
- **The top catcher's share of home starts:** median 62% (2024) and 58% (2025).
- **The build's catcher** is in his side's first nine batters in 93.9% of 2024 team-games and 94.4% of 2025's. In the other 6% or so, the build recorded a catcher who was not in the starting nine.

## 5. Procedure
**The run:**
- Test season 2026 only.
- Blend walk-forward as in the screen: the 12-model blend, retraining every 7 days, `estimated_pa` basis, `BTS_LGBM_DETERMINISTIC=1`, the seed as `BTS_LGBM_RANDOM_STATE`, and the screen's feature settings, re-checked against production's at admission.
- Labels rebuilt with the resumed portion excluded (BTS scoring).

**Seeds:** the screen's ten, positions 0–9 of `canonical-n10.json`, with one run per seed.
- They run in order, one at a time, through the C1 launcher. Seed 1 runs first and is checked for cost.
- No launch from 00:45 to 03:10 box clock, and the manager gets a heads-up before each launch.

**Sealing:**
- Outcomes stay sealed until all ten seeds have reconciled.
- Opening them is announced to the manager first, with the ledger total and every TERMINAL/RECONCILED record, as in stage two.
- There is no early stop on outcomes, only on cost (§7).

## 6. Dispositions (fixed now): A-posted against the baseline, all ten seeds
**Quantities:**
- d = a seed's 2026 P@1 delta (A-posted − baseline).
- m = the mean of the ten d.
- L = the 10th percentile of m under day resampling:
  - Draw the 2026 test days with replacement 10,000 times, using numpy's `default_rng(20261009)`.
  - Each draw applies to all ten seeds jointly, and the ten-seed mean delta is recomputed each time.
- **The day set** must be identical across every arm and seed. If it is not, the result is incomplete.

**Dispositions:**
- **positive** requires all three:
  - m ≥ +0.3pp;
  - L > 0;
  - d > 0 on at least 6 of 10 seeds.
- **negative:** m ≤ 0.
- **inconclusive:** anything else.
- **incomplete:** anything other than exactly the ten registered seeds, each validated.

**Why L.** In a single season, most of the uncertainty comes from which days the season happened to have, not from the seed. On 2024 and 2025, A and the baseline chose the same top batter on 81% and 75% of days. Day resampling of the ten-seed mean gave a standard error of 1.03pp (2024) and 1.43pp (2025); one seed alone gave about 2pp. Seeds do not reduce this day-level noise.

**What this rule can resolve** (reasoning from 2024–2025's noise, not measured on 2026):
- If A's true effect equals the screen's +1.95pp, a positive has about a 6-in-10 chance.
- At a true +1.0pp, about 1 in 3.
- At zero, about 1 in 10.

**A-projected** has no weight in A's disposition. When A-posted is positive, it decides the production fallback:
- If its m against the baseline is ≥ 0, a projected catcher is usable before lineups post.
- If its m is < 0, the note recommends using A's catcher value only once the opposing lineup is posted. The missing-value fallback would then need its own check before D7.

**Streak metrics** (`mean_max_streak`, exact P(57) and the replay longest streak) are reported per seed for both A arms and carry no weight in the disposition. The note states first whether they moved against P@1, for two reasons:
- D1's objective is the longest streak.
- The screen's streak simulation fell on 8 of 10 seeds under A.

## 7. Compute, budgets and stops
**Estimate** (reasoning, not measured on 2026): about 2.75 CPU-h per 2026 walk-forward.
- 2025's walk-forwards averaged 2.38, and 2024's averaged 2.01. 2026 trains on one more season again.
- That gives about 8.3 CPU-h per seed for three arms, and about 83 for ten seeds.
- For comparison: baseline and A-posted alone are about 55; adding B would make it about 110.

**Proposed budgets** (Eric's to set, in his own typed words, before the first launch):
- 12 CPU-h per seed.
- A first-walk-forward stop at 4.1 CPU-h (1.5 × the estimate).
- A total allowance of 100 CPU-h.

The box is fixed-price, so the cost is $0 of new spend.

## 8. Process and reporting (as stages one and two)
1. This design is reviewed by a fresh Codex session (memories off) and frozen at its reviewed commit.
2. The code is written test-first: the catcher-table builder, the walk-forward hook, and the 2026 admission and aggregate. It gets a Codex code review (memories off) and is frozen.
3. X-37 is recorded.
4. Seed 1 runs, then its cost is checked, then seeds 2–10 run.
5. The aggregate runs, then the results note is written.
6. An independent acceptance follows: a fresh Codex session, memories off, Part 1 blind.
7. The result goes to Eric in the lead's pane, plain language first, then to the manager.

## 9. What a result can support
**A positive** is out-of-sample evidence on one season. With Eric's exception (above), it can bring A to D7 for 2027 as a production change, which gets its own review and deploy gating. The change has two parts:
- a catcher lookup keyed by catcher id;
- the opposing catcher on each slate slot, taken from the posted lineup, with the projected rule chosen by §6.

**Inconclusive or negative:** the baseline stays. A 2027 shadow remains possible.

## 10. Limits
- **One season.** Day resampling describes 2026's day-to-day noise. It cannot show the effect carries over to 2027.
- **The posted catcher is approximated** by the final boxscore's starting nine. A catcher scratched after the lineup was posted counts as posted. No pregame capture of lineup identities exists: the lineup collector records only the times lineups were confirmed, and pick files hold only the pick.
- **Production picks are mixed.** A pick commits once its own side's lineup is confirmed. The opposing lineup and later games' lineups may still be projected then. The two A arms bound that mix; they do not reproduce it.
- **Only the catcher is changed.** Everything else about the walk-forward is as in the screen: it still uses each day's realized PA rows (actual pitchers, actual lineups).
- **Production's lookups lag the backtest by one appearance** for every feature (the known M3 serving-staleness gap).
- **The seeds** are the screen's ten and keep its limits (outcome-ranked source set; 32-bit wraparound of five seeds).
- **2026 is not untouched in general.** W1.2 read the production recipe's 2026 walk-forward level. The A-versus-baseline contrast is what remains unread.

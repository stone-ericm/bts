# Optimization Ideas (Parking Lot)

## Known Edge Cases for `bts predict`

### Opener vs starter
Some teams use a reliever ("opener") for the first 1-2 innings, then a bulk pitcher for the rest. Our prediction grabs pitcher features from the first PA's matchup, which would be the opener — wrong for 3 out of 4 PAs. The bulk pitcher's features are what matter.

**Detection**: Check if the scheduled "starter" has a reliever profile (low innings/appearance, high appearances, bullpen usage pattern). Or pull from team-specific opener tendency data.

**Impact on backtesting**: None — backtesting uses actual PA-level pitcher_id.

### Players on the Injured List
A player on the IL can't play and shouldn't be picked. A player just activated from the IL has stale rolling features (batter_hr_7g from weeks ago) and high days_rest. The model has few training examples with 15+ day gaps, so predictions for IL returns are unreliable.

**Fix**: Check roster status via MLB API before generating picks. Flag any player with days_rest > 7 as unreliable. Consider a minimum recency threshold for rolling features.

### Other edge cases to handle
- **Postponed/suspended games**: Pick wasted if game is rained out
- **Late scratches**: Player removed from lineup after pick is locked
- **Doubleheaders**: Player could play in both games (pick applies to first)
- **Interleague DH rules**: Universal DH since 2022, so no longer an issue

## Scheduler Edge Cases

### Singleton slates get zero pre-start lineup checks (observed 2026-07-16; backlogged 2026-07-27)
`compute_run_times` derives every lineup check from game starts (start − `lineup_check_offset_min`,
60 on the box), computed once from the 10:00 schedule fetch. Two compounding gaps, both live on
2026-07-16 (the 1-game post-ASB slate, NYM@PHI):

1. **One game → one check.** The whole day's plan was a single check at 18:10 ET; the scheduler
   slept 10:00→18:10 with no earlier anchor (no other games' checks) and no intermediate fallback
   tick, so nothing could deliver before that moment.
2. **The plan is static.** The 10:00 fetch saw the original 23:10Z (19:10 ET) start; MLB moved the
   game up to 22:10Z (18:10 ET) afterward. The move consumed the entire 60-minute margin — the lone
   check fired exactly at first pitch, classified `game_started_or_final`, and the day locked
   undelivered.

Outcome: preview pick (Turner, p=0.675, PROJECTED) never delivered → contest Pass, streak untouched,
and correctly no entry-checker nag. Effectively silent: the only same-day signal was the incidental
log-only `calibration_drift` WARN (top-1 p below the 0.70 floor). Benign that day because the pick
was weak anyway, but a strong pick on a future 1-game slate would be passed the same way with no
alert.

**Fix shape (small):** when the slate has <2 games (or generally for the pick's game), guarantee at
least one wake before the T−`fallback_deadline_min` cutoff — e.g. checks at lineup-posting-time
estimates (`data/lineup_posting_times`) and/or fixed T−120/T−90 offsets — and re-fetch the schedule
at each wake so a moved-up start re-anchors the remaining plan. Compose with the existing
deferred-fallback and DD both-slots-confirmed gates rather than bypassing them; consider a loud
missed-delivery alert when a day ends `game_started_or_final` with a never-delivered pick.

**Frequency/urgency:** 1-game slates are rare (post-ASB opener, odd makeup days) — low urgency,
small blast radius, but silent when it hits.

## Feature Ideas
- Batter H2H history vs specific pitcher (sparse but might add on top of archetype)
- Pitcher recent workload (days since last start, innings in last 7/14 days)
- Batter launch speed/angle rolling averages (batted ball quality trends)
- Time-of-day effects (day games vs night games — dayNight field in game feed)
- Batter career PA count (experience proxy)
- Opposing team bullpen quality (for later PAs in game aggregation)

## Model Ideas
- Optuna hyperparameter search (systematic instead of manual grid)
- CatBoost as alternative to LightGBM (handles categoricals natively)
- Stacked ensemble: LightGBM predictions as features for a second-stage model
- Probability calibration post-hoc (isotonic on held-out fold)

## Architecture Ideas
- Pre-game PA estimation: use lineup + pitcher history to estimate number of PAs per batter
- Separate models for first PA vs later PAs (different pitcher fatigue context)
- Opponent-adjusted features (batter stats adjusted for quality of pitchers faced)

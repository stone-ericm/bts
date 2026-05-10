# Leaderboard clue audit

**Date**: 2026-05-09 ET
**Scope**: exploratory, read-only analysis of captured public BTS leaderboard data.
**Script**: `scripts/analyze_leaderboard_clues.py`
**Prod input**: `/home/bts/projects/bts/data/leaderboard/`

This is not a dashboard feature review. The question is whether public
leaderboard behavior contains clues that can improve our picks.

## Headline

This is a **post-hoc, survivorship-biased, exploratory** result. The
leaderboard data is worth exploiting, but only under a prospective protocol.
No production parameter, pick rule, threshold, or strategy edit is supported by
this memo alone.

On the current post-hoc corpus, tracked-user consensus strongly outperformed our
realized picks on overlapping historical date-slot units:

| Comparison | n | Hit rate |
|---|---:|---:|
| Our realized picks on overlap | 75 | 0.680 |
| All-tracked consensus on same units | 75 | 0.813 |
| Disagreement-only: our picks | 72 | 0.667 |
| Disagreement-only: tracked consensus | 72 | 0.806 |

The comparison unit here is one `(pick_date, pick_number)` unit: our production
pick for that date/slot versus the modal tracked-user pick for the same
date/slot. It is not a user-level sample, and units are correlated by slate,
candidate pool, and season phase.

Agreement was only 0.040, so the leaderboard surface is not just rediscovering
our current policy. It is pointing at a genuinely different policy family.

This is a clue, not a verdict. The corpus is selected from users who were
visible enough to be tracked in the leaderboard scrape and then backfilled
through their historical pick logs. That creates both selection bias
(`tracked users` are not all entrants) and survivor/right-truncation bias
(users who missed before entering or remaining visible are underrepresented).
Those biases can inflate apparent consensus quality. We should use this as a
policy-mining and falsification surface, not as direct evidence that "copy
consensus" is optimal.

## Data Inventory

Read-only prod inventory:

| Item | Count |
|---|---:|
| `user_picks/*.parquet` files | 577 |
| Empty user-pick files | 62 |
| Non-empty users | 515 |
| Raw appended rows | 234,720 |
| Deduplicated pick rows | 28,568 |
| Resolved hit/not-hit rows | 27,978 |
| Pick dates | 45 |
| Date range | 2026-03-25 to 2026-05-08 |
| Leaderboard snapshots | 9 |
| Latest snapshot | 2026-05-09.parquet |

Deduplication used `username + pick_date + pick_number`, keeping the latest
observation. This matters because the existing helper
`read_user_picks(..., dedupe="latest_per_pick_date")` dedupes by date only and
can drop the second pick on double-pick days.

## Aggregate Behavior

Resolved tracked-user hit rates:

| Cohort | Picks | Users | Hit rate | Slot 1 | Slot 2 |
|---|---:|---:|---:|---:|---:|
| All tracked users | 27,978 | 515 | 0.758 | 0.760 | 0.754 |
| Latest active top-100 users | 5,009 | 100 | 0.764 | 0.782 | 0.737 |

Double-pick behavior:

| Metric | Value |
|---|---:|
| User-days with pick 2 | 12,051 |
| Resolved double-pick user-days | 11,569 |
| Share of user-days with pick 2 | 0.730 |
| Both-hit rate on resolved double-pick days | 0.577 |
| Pick 1 hit rate when double-picking | 0.736 |
| Pick 2 hit rate when double-picking | 0.753 |

The public leaderboard population double-picks heavily. That does not mean we
should always double-pick, but it argues against assuming conservative
single-pick behavior is common among successful entrants.

## Consensus Surface

All tracked users, modal batter by date/slot:

| Units | Valid units | Consensus hit rate | Avg top share | Median top share | Max top share |
|---:|---:|---:|---:|---:|---:|
| 90 | 89 | 0.787 | 0.190 | 0.174 | 0.548 |

Latest active top-100 users, modal batter by date/slot:

| Units | Valid units | Consensus hit rate | Avg top share | Median top share | Max top share |
|---:|---:|---:|---:|---:|---:|
| 90 | 90 | 0.811 | 0.190 | 0.170 | 0.490 |

Recent all-tracked consensus examples:

| Date | Slot | Consensus | Share | Result |
|---|---:|---|---:|---|
| 2026-05-01 | 1 | Vladimir Guerrero Jr. | 0.242 | not_hit |
| 2026-05-01 | 2 | Ozzie Albies | 0.220 | hit |
| 2026-05-02 | 1 | Carlos Cortes | 0.134 | hit |
| 2026-05-02 | 2 | Bobby Witt Jr. | 0.116 | hit |
| 2026-05-03 | 1 | Ozzie Albies | 0.174 | hit |
| 2026-05-03 | 2 | Ozzie Albies | 0.199 | hit |
| 2026-05-04 | 1 | Juan Soto | 0.101 | not_hit |
| 2026-05-04 | 2 | Bobby Witt Jr. | 0.132 | hit |
| 2026-05-05 | 1 | Nico Hoerner | 0.236 | hit |
| 2026-05-05 | 2 | Nico Hoerner | 0.169 | hit |
| 2026-05-06 | 1 | Yandy Diaz | 0.149 | hit |
| 2026-05-06 | 2 | Juan Soto | 0.211 | hit |
| 2026-05-07 | 1 | Bobby Witt Jr. | 0.290 | hit |
| 2026-05-07 | 2 | Miguel Andujar | 0.198 | hit |
| 2026-05-08 | 1 | Vladimir Guerrero Jr. | 0.153 | not_hit |
| 2026-05-08 | 2 | Bobby Witt Jr. | 0.161 | hit |

## Batter Concentration

Most-picked resolved batters among tracked users:

| Batter | Picks | Hit rate | Users |
|---|---:|---:|---:|
| Vladimir Guerrero Jr. | 2,139 | 0.735 | 462 |
| Shohei Ohtani | 1,567 | 0.757 | 446 |
| Bobby Witt Jr. | 1,376 | 0.845 | 421 |
| Yordan Alvarez | 1,330 | 0.729 | 411 |
| Aaron Judge | 1,302 | 0.677 | 407 |
| Luis Arraez | 1,121 | 0.831 | 398 |
| Freddie Freeman | 907 | 0.840 | 371 |
| Juan Soto | 781 | 0.849 | 356 |
| Yandy Diaz | 780 | 0.833 | 308 |
| Nico Hoerner | 644 | 0.842 | 304 |

There is a visible "known safe hitter" policy family: top users repeatedly pick
elite contact/production names, and our production policy often disagrees. The
right next question is not whether those names are intrinsically better, but
whether our candidate ranking underrates the same batter class on days when
leaderboard users concentrate there.

## Current Interpretation

1. Leaderboard consensus is a promising auxiliary signal.
2. It should not replace the model from this evidence alone because the current
   analysis is post-hoc and survivorship-biased.
3. The immediate value is a **candidate audit feature**: for each daily
   candidate, record consensus share / top-user pick share / whether it is the
   public consensus pick, then compare that to our model rank and realized result.
4. We need an availability audit before using same-day consensus operationally:
   determine whether current-day user picks are visible before our lock deadline,
   or only after lock/result. If not visible pre-lock, this remains a historical
   policy-mining signal, not a same-day input.
5. The existing dashboard helper's date-only dedupe is not suitable for serious
   leaderboard analysis because it can discard double-down picks.
6. Any subgroup reads from this memo, including slot, batter class, top-user
   subset, consensus-share tier, or double-pick behavior, are diagnostic only.
   A confirmatory subgroup claim needs FDR control or a separate
   pre-registered test family.

## Proposed Prospective Protocol

Treat this memo as exploratory. Do not make a pick-rule change from it alone.

Next, pre-register and run a prospective leaderboard-vs-model comparison.

1. For each future slate, before our pick lock, capture any visible leaderboard
   pick data and candidate artifacts. If current-day picks are not visible
   before lock, label the day unusable for same-day operational inference.
2. Define a fixed forward cohort at the first pre-lock capture of the forward
   window. Users stay in the cohort even if they later miss, disappear, or fall
   down the leaderboard. This does not eliminate selection bias, but it avoids
   conditioning the forward test on later survival.
3. For every candidate, attach:
   - all-tracked consensus share,
   - fixed-cohort consensus share,
   - rank among public picks,
   - whether the candidate is public consensus,
   - our model rank and `p_game_hit`.
4. Define primary comparison units as resolved `(pick_date, pick_number)` units
   where our pick disagrees with pre-lock public consensus.
5. Primary estimand: `mean(consensus_hit - our_hit)` over future resolved
   disagreement units, with slate/day-block uncertainty rather than treating
   correlated same-day units as independent.
6. Secondary estimand: whether consensus share improves candidate ranking after
   conditioning on our `p_game_hit`.
7. Store forward artifacts under `data/validation/leaderboard_clue_audit_<DATE>.json`
   or a matching parquet path. Do not write to production state or scheduler
   paths.
8. Re-evaluate on a fixed cadence: first at 30 future resolved disagreement
   date-slot units, then monthly or at season end. Re-running the exploratory
   script tomorrow with one more day of data is not a significance test.
9. Falsification criterion: if the first forward evaluation has
   `mean(consensus_hit - our_hit) <= 0`, or an uncertainty interval that clearly
   includes zero with no mechanism found in the candidate join, reject the
   consensus-edge hypothesis for production use until more evidence is
   pre-registered.
10. No production policy edit until either:
   - at least 30 future resolved disagreement date-slot units support a positive
     consensus edge under the fixed-cohort protocol, or
   - a candidate-join audit shows a clear, reproducible mechanism that our model
     is missing and that mechanism passes its own validation gate.

## Implementation

`scripts/leaderboard_candidate_join_audit.py` implements the forward artifact
surface for this protocol.

Inputs:

- `--artifact-dir`: frozen live candidate artifact directory containing
  `manifest.json` and paired production/candidate ranked profiles.
- `--leaderboard-dir`: captured public leaderboard parquet store.
- `--decision-cutoff-iso`: latest leaderboard observation timestamp allowed
  into the join. This is the pre-lock visibility gate.
- `--cohort-as-of-iso` or `--cohort-users-json`: fixed forward cohort source.

Outputs:

- JSON report, intended for
  `data/validation/leaderboard_clue_audit_<DATE>.json`.
- Joined candidate profile parquet, defaulting to the same path with
  `.joined.parquet`.

The implementation fails loud on malformed leaderboard parquet inputs. It
deduplicates public picks by `username + pick_date + pick_number`, so double-pick
days are preserved.

Important limitation: leaderboard user-pick rows currently include `batter_id`
but not `game_pk`. Candidate popularity features therefore join on
`date + batter_id`, not `date + batter_id + game_pk`. This is acceptable for
candidate-level clue mining, but doubleheader/game-context ambiguity must remain
in the artifact caveats until an additional source resolves it.

The comparison summary uses the fixed cohort as the confirmatory surface and
keeps all-tracked output diagnostic. Day-block uncertainty uses a contiguous
block bootstrap with `expected_block_length=7` and default `seed=20260509`.
The block length matches the live-forward comparison rule; the seed is recorded
in each JSON artifact and can be changed only by future pre-registration.

Initial smoke run used the 2026-05-09 live-forward artifact and wrote only to
`/tmp` on prod:

```
ssh -o BatchMode=yes -o ConnectTimeout=8 bts-hetzner \
  'cd /home/bts/projects/bts && /home/bts/.local/bin/uv run python - \
    --artifact-dir data/validation/decision_weighted_lgbm_v0_live_forward/2026-05-09 \
    --leaderboard-dir data/leaderboard \
    --output /tmp/leaderboard_candidate_join_2026-05-09.json \
    --joined-output /tmp/leaderboard_candidate_join_2026-05-09.joined.parquet \
    --decision-cutoff-iso 2026-05-09T18:15:12.096614+00:00 \
    --dates 2026-05-09 \
    --n-bootstrap 0' \
  < scripts/leaderboard_candidate_join_audit.py
```

The smoke artifact produced `n=0` fixed-cohort comparison units for 2026-05-09,
which is a valid negative availability result: the captured leaderboard corpus
had no current-date consensus rows visible for the artifact date. That supports
the need for the pre-lock visibility gate before using leaderboard consensus
operationally.

## Reproducibility

The exploratory and prospective scripts were syntax-checked locally:

```
uv run python -m py_compile \
  scripts/analyze_leaderboard_clues.py \
  scripts/leaderboard_candidate_join_audit.py
uv run python scripts/analyze_leaderboard_clues.py --help
uv run python scripts/leaderboard_candidate_join_audit.py --help
```

Prospective join tests:

```
uv run pytest tests/scripts/test_leaderboard_candidate_join_audit.py -q
```

The exploratory prod analysis was run read-only via SSH:

```
ssh -o BatchMode=yes -o ConnectTimeout=8 bts-hetzner \
  'cd /home/bts/projects/bts && /home/bts/.local/bin/uv run python -' \
  < scripts/analyze_leaderboard_clues.py
```

No prod files were written.

# Item 2: Official Scoring Retroactive Changes

**Verdict:** NOT HANDLED

## Analysis

### How `check_hit` works

`check_hit` in `src/bts/picks.py` makes a single API call to:

```
GET https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live
```

It first checks `gameData.status.abstractGameCode == "F"` (Final). If the game is not
Final, it returns `None` and the streak is left unchanged. If Final, it reads
`liveData.boxscore.teams[side].players[ID{batter_id}].stats.batting.hits` and returns
`hits > 0`.

Importantly, it reads the current boxscore hits value — not an original scoring decision.
The MLB Stats API reflects the official scorer's current ruling, including any retroactive
corrections. So if a hit was changed to an error after the game ended, a subsequent call
to this endpoint would return `False` instead of `True`.

### Does it re-verify previous days?

No. The `check-results` CLI command (cli.py, line 475) takes a `--date` argument and
checks exactly that one date. There is no loop over recent dates, no re-check of already-
resolved picks, and no comparison of a stored result against a fresh API response.

Once a result (`"hit"` or `"miss"`) is written to `data/picks/{date}.json` and the streak
is updated via `update_streak`, that result is permanent from the system's perspective.

### The cron timing gap

Per ARCHITECTURE.md, `check-results` runs via cron at 1am ET. The orchestrator
(`orchestrator.py`) does not call `check-results` — it handles predictions and posting
only. Results checking is entirely the responsibility of the CLI command.

MLB's official scoring review window runs until 6pm ET the day after the game. For a
game ending at 11pm ET Tuesday, the system checks results at 1am ET Wednesday (roughly
2 hours post-game) and writes the result. A scorer's decision could be changed any time
before 6pm ET Wednesday — a 17-hour window after the check runs — and the system would
never know.

### Time window breakdown

```
Game ends:        ~11pm ET Day 0
check-results:     1am ET Day 1  ← result written, streak updated
Scoring review:   6pm ET Day 1  ← MLB official deadline for changes
Gap:              ~17 hours after check-results during which a change could occur
```

## Risk Assessment

### How often do scoring changes happen?

Retroactive hit-to-error changes are rare but not negligible. MLB data shows roughly
50-100 official scoring changes per season league-wide, most of them fielding plays on
borderline grounders or line drives. The probability that any single pick involves a
retroactive change is low (maybe 0.1-0.5% of picked games). Over a full season of ~100
pick-days, the expected exposure is roughly 0.1-0.5 incidents per season.

### Could this affect a live streak?

Yes, and in both directions:

- **False hit (worse case)**: The system logs a hit at 1am; the scorer reverses it to an
  error by 6pm. The streak was incorrectly extended. If the actual BTS account is
  updated by MLB's backend with the corrected ruling, the in-game streak and our tracked
  streak diverge. We could believe streak=5 while BTS says streak=0.

- **False miss**: A no-hit ruling at 1am gets corrected to a hit. The system reset the
  streak to 0; the actual BTS streak survived. This is operationally recoverable (manual
  fix) but damaging to streak integrity tracking.

The false-hit direction is the more dangerous one at high streaks, where a divergence
between our tracker and the BTS game would cause incorrect MDP decisions on subsequent
days.

### Realistic exposure in 2026

Current streak is 4. At low streaks, a false reversal has minimal compounding impact.
At streaks 10+, a 1-day discrepancy in the tracker compounds through every future
skip/double-down decision. The expected cost is low this season, but grows quadratically
with streak length because of the exponential nature of P(57).

## Recommendation

No implementation needed now — the expected frequency is ~0.1-0.5 events/season and the
current streak is 4. Document the gap and add a manual check trigger.

If implemented, the fix would look like this:

1. **Deferred finalization**: Do not write `result` to the pick file until a 24-hour
   hold period has passed. At 1am, write a `result_preliminary` field but leave `result`
   as `None`. Run a second cron pass at 7pm ET the following day that finalizes any
   pending results.

2. **Alternatively, re-check window**: Modify `check-results` to also re-verify any
   pick files where `result` is set but `run_time` was less than 20 hours ago, comparing
   the stored result against a fresh API call. If the results differ, log a warning,
   update the result, and recalculate the streak from that day forward.

3. **Minimum viable guard**: Before writing `result`, log the raw `hits` value from the
   API alongside `result` in the pick file. This creates an audit trail and makes manual
   verification after a scoring change straightforward. Cost: two extra JSON fields.

The minimum viable guard (option 3) is the right move now — zero logic change, full
auditability. The deferred finalization (option 1) is the correct fix if a high streak
(20+) makes the risk unacceptable.

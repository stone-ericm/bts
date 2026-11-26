# Kickoff B4: three findings from the second reviewer (Gemini 4 Carbon), each reproduced by the lead with its probe

Report: `/Users/eric/projects/bts/docs/audit/2026-11-14-season-rollover-carbon-r1.md` (B2, B4, B5 and "False greens"; its B3 is the loop-variable problem you have already fixed). Its probe, which I reran against the merged candidate, is at `/Users/eric/projects/bts-agy-sr/.codex-review/sr-review/scratch/probe_s6.py` (read-only for you; run it with `PYTHONPATH=.` from a checkout if you want to see it).

## 1. An extra attempt has room for one cascade, not two
| State | On your branch | Required |
|---|---|---|
| an extra attempt starts at a start's fallback deadline and its check returns a pick that the confirmation gate does not lock (`should_post` false, for example another game still projected), and that pick's own fallback deadline is already reached | the in-loop fallback at once runs a second refresh cascade; the probe shows cascades at 18:35 and 18:51 for 19:10 starts, the cutoff 19:05 overrun, the pick refused at 19:07 and lost, then the day-end alert | the pick from the attempt's own cascade is delivered through the fallback delivery with no second cascade (the probe's patched run delivers at 18:51) |
| the same, but the returned pick's own fallback deadline is still ahead (the pick is in a later game) | (existing handling) | existing handling, unchanged |

The reviewer's smallest fix: `run_single_check` computes the lock decision but does not return it, so the in-loop fallback's `result.get("lock_decision")` is always `None` and its "reuse this check's fresh cascade decision" branch never runs. Returning it makes that branch live. That branch then also becomes live for PLANNED checks, which is a behaviour change outside S6. Decide with evidence and say which you chose:
- (a) enable it for every check, if you can show that it is what the existing code intends (read the 2026-08-30 notes in `CLAUDE.md` and `docs/audit/2026-08-30-late-pick-delivery.md`) and that every existing scheduler test passes unchanged; list each existing day-level scenario whose outcome changes, if any; or
- (b) limit the reuse to extra attempts.
If (a) changes any existing scenario's outcome, choose (b).

## 2. Games that can no longer be played get no extra attempt and no alert
| State | Base `main` | Your branch | Required |
|---|---|---|---|
| every game of the day is postponed or cancelled after the morning fetch | three planned checks, no DM; `bts day-outcome` reads `no_games` | an extra cascade at 18:35 and a CRITICAL day-end alert | no extra attempt, no alert |
| some games void, some playable | | | extra attempts only for starts that still have at least one playable game; the alert only if at least one of today's games is still playable when the day's attempts end |
| the status read fails | | | treat the games as playable (fail open, as the delivery guard does) |

Use the status source the delivery guard uses (`_delivery_game_statuses` and the void test in `picks`), at most one status read per extra attempt and one before the alert. `tests/conftest.py` already keeps that read off the network.

## 3. The alert is dated by the slate, not the wall clock
`_alert_missed_pick` calls `dispatch_dm_for_health_alerts` without `now_et_date`, so its once-per-day dedupe keys on the wall-clock date. After midnight a second DM goes out for the same slate, and the next day's real alert is suppressed (probe 4A). Pass the slate's date, as `health/runner.py` does. This applies to the old missed-pick alert as well as the new one; say whether the old one could already misfire this way at the base.

## Tests
- RED first for each row above, driving the real `run_day`.
- Every existing extra-attempt test arranges `should_post=True`; add the `should_post=False` cases of item 1 (both rows).
- `tests/test_cold_start_integration.py`, the control test: the reviewer found the feed-time assertion weakened to a subset check. Assert the exact set of feed requests before first pitch again, and treat result-polling requests after delivery separately.
- Rerun every scheduler test file (your B3 list) and report the totals.
- Restate the worst-case number of extra cascades after these changes.

## Deliverable
`.codex-review/sr/handover-b4.md`, under 50 lines; `git diff --stat` against your new base commit (see `git log -1`). Do not commit. Print `SRB4-<first 8 hex characters of the sha256 of the handover file>` and then `DONE`.

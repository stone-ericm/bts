# STOP: change set G, the monotone skip guard (R61), after its second round (2026-11-25 00:05, `bts-wr-fresh10`)

**Not merged.** Branch `fix/wr-g` (pushed): code `d6141728` (round 1) and `cfc62745` (round 2); contract `kickoff-fix-g.md` version 2 with amendments 1 to 3 and both rounds recorded; contract check `contract-check-g/`; reviews `review-fix-g-codex/`, `review-fix-g-carbon/`, `review-fix-g-codex-r2/`. `main`'s `src/` is unchanged: the shipped table still skips its top bin in 1,511 states.

## In plain words
The guard itself, the thing Eric approved, was right in both rounds: four independent reviews found exactly the 1,511 intended states changing, to the intended action, on every path that makes a pick, with nothing else changing and the season walk as predicted. **Both rounds blocked on `decision.json`, the day's record of what was committed, not on which pick is made.**
- Round 1: the first author added a rule nobody asked for (never rewrite an existing record), which left the wrong record after five real commit sequences. Removed in round 2.
- Round 2: two cases where the record names a batter that was not committed. (1) A single is recorded with the runner-up in its second slot. (2) A saved pick sent by the last fallback, after a refresh that looked at a different slate, is recorded with that refresh's batter and no guard.

## What the session found by reading the base (not run)
Both round-2 cases are how the base's record writer already behaves, guard or no guard: `_commit_decision_for_pick` at `f83e0d0b` copies `selection.double_candidate` into `double_down` for every fresh pick (CLAUDE.md, fix set B, W8: "the decision record still stores the runner-up in `double_down` on a single"), and takes its metadata from whatever decision the cycle's selection holds. They became blockable because G's contract (the contract check's text) requires the guard, the action and the slots to be equal between a fresh and a recovered record, and because round 2's item 2 made the recovered record use the saved pick's slots. The reviewer's fix for both: the record's slots always come from the pick being committed, and a selection's decision is used only when that selection produced that pick.

## Ways forward (the lead rules: R57; no override of a second-round BLOCK: R51)
1. **A small follow-up, G2b, with a fresh verdict on the changed lines only** (recommended). In `_commit_decision_for_pick`: `primary` and `double_down` from the committed pick's own slots in every case; the selection's decision metadata, streak and saver only when the selection made this pick (`selection.decision is not None and selection.pick_result.daily is daily`, the test change set 5a already uses), otherwise the saved pick's `policy_decision`. Plus E1 and the three notes as tests. Cost if wrong: it changes what every day's record holds in `double_down` on a single (a known oddity, W8), so every reader of that field must be checked (`day-outcome` already ignores it on a single; the skip-policy shadow uses `second_candidate` on skip records, a different field); a third review could find another record path.
2. **Eric accepts G as it stands** (the picks are right; the two record cases pre-date G) and it merges on a green gate. Cost: on a guarded single with doubles disabled, or the rare restarted-fallback case, `decision.json` names a batter that was not committed; grading reads the pick file, so the streak is not affected, but `bts day-outcome`, the census and the season ledger read the record.
3. **Leave `main` as it is until the record writer is fixed as its own change set**, then land the guard on top. Cost: the season could open with the top-bin skip (rare in play: streak 41 or more, or the last 30 days with a bin-1 candidate).

## Also for the lead
- No full gate has run on round 2's code. Round 1's gate (10,804 passed) found one flipped pin the contract check had missed; amendment 2 item 7 fixed it.
- Carbon's round-2 review was still running when this was written; it is archived to `review-fix-g-carbon-r2/` when it reports.
- H (R62) was not started.

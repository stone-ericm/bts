# Kickoff: change set G, the monotone skip guard (version 1, 2026-11-24 22:10, session `bts-wr-fresh10`; drafted by `bts-wr-fresh9`)

Ruling R61 (`FINDINGS.md`); Eric approved the policy change on 2026-11-24: **never skip a candidate when the table would play a weaker one in the same state.** Tier A (it changes which pick is made). Investigation: `docs/audit/2026-11-24-policy-top-bin/README.md`.

**Base:** branch `fix/wr-g`, cut from `main` at `f83e0d0b` (the season walk is on `main`; change sets E and Ea, the operator texts, are STOPPED and not on `main`, so the skip notice's wording is `main`'s). Tests first (RED on the base), then the change, ONE commit, no push.

## The rule
In the reach-57 regime with the base table loaded, the table is looked up as today: state (streak `s` = min(streak, 56), days left `d` = min(days, season length), saver `v`) and the candidate's quality bin `q` from the stored boundaries. If the table's action for `(s, d, v, q)` is skip AND some lower bin `j < q` has a non-skip action in the same `(s, d, v)`, the policy's action is the table's action for the HIGHEST such `j` (single or double, whatever that bin holds). In every other case the action is the table's own.

Nothing else changes:
- not the artifact (`data/models/mdp_policy.npz`, sha `66d15471…`, untouched; the tail policy's binding to that sha stays valid) and not the tail artifact;
- not the tail regime (`OBJECTIVE_TAIL`: `_tail_action` and the forced rule are untouched), not the heuristic path (no base table), not the early returns of the lookup (streak 57 or more, no days left: skip);
- not a single against a double anywhere the table already plays; not the `allow_double` clamp or the "double needs a different-game second pick" clamp, which apply to the guarded action exactly as they apply to a table action;
- not `bts.simulate.mdp.lookup_action`: it keeps returning the table's own action, so every offline evaluator of a table (`simulate/exact.py`, `validate/ope.py`, `pooled_policy.py`, the audit scripts) is unchanged. The guard is a new function used by the production decision only.

## Where
1. `bts.simulate.mdp`: a pure function `lookup_action_guarded(policy_table, boundaries, streak, days_remaining, saver, top_pick_confidence, season_length=180) -> tuple[str, dict | None]`. It shares the state and bin computation with `lookup_action` (one private helper used by both, so the two cannot bin differently). The second element is None when the action is the table's own, else `{"rule": "monotone_skip_v1", "bin": q, "from_bin": j, "table_action": "skip"}` (plain ints and strs).
2. `bts.strategy._mdp_action_from` calls it and returns `(action, guard)`; its callers are `resolve_policy_decision` and `effective_pick_bar` (which needs only the action).
3. `PolicyDecision` gains a LAST field `guard: dict | None = None`, set from the lookup in the reach-57 table branch and None everywhere else (tail, heuristic). `policy_action` is the guarded action (the policy's action before the clamps), as today it is the table's.
4. The decision record: `daily_decision.write_decision` gains a keyword `policy_guard=None`, written as the key `policy_guard` (None on every record the guard did not decide). The schema tag stays `bts_daily_decision_v3` (readers ignore unknown keys; `parse_decision` is untouched). `scheduler._selection_decision_meta` carries it from the cycle's `PolicyDecision`, or from the pick file's `policy_decision` (which is `asdict(decision)` and so carries `guard` by itself) on the cached-fallback and restart paths; every `write_decision` call site that passes the meta passes it through. A skip record never carries a guard (a guarded action is never a skip).
5. The journal: `scheduler._log_policy_decision` prints one line when the guard decided: `  POLICY GUARD — the table skips quality bin {q} at streak {s} with {d} days left (saver {available|not available}) while bin {j} plays: playing as bin {j} ({action}).` No DM text changes (the pick message is the pick message).

## The 1,511 affected states
`docs/reviews/2026-11-18-whole-range/contract-g/affected-states.csv` (sha256 `77ec5404…`, made by `contract-g/enumerate_states.py` from the artifact alone; 1,511 rows of 20,996 states, exactly one affected bin per state): 1,156 states where bin 4 skips and inherits bin 3's single; 355 where bin 1 skips and inherits bin 0's action (171 single, 184 double). Streaks 1 to 56; none at streak 57 or with no days left; 954 with more than 30 days left, 557 with 30 or fewer.

**Must change (a test reads the CSV):** for every row, the guarded lookup at a probability inside bin `bin` returns `guarded_action` with a guard naming `bin` and `from_bin`; and `resolve_policy_decision` on a 2027 date with that many days left (where one exists: days 1 to 180), that streak and saver, a primary probability in that bin and an executable second pick, returns that action with `source == "mdp"`, `objective == "reach57"` and the same guard, and with `has_diff_game=False` a guarded double becomes a single.

## Must still happen
1. **Every unaffected cell returns exactly the table's action:** over all 58 × 181 × 2 × 5 = 104,980 cells, at three probabilities per bin (its lower edge, its midpoint, just under its upper edge; for the end bins 0.0 and 0.999), `lookup_action_guarded` equals `lookup_action` with guard None, except the 1,511 cells of the CSV. `lookup_action` itself is equal to the base on all of them (run the base's function from `git show <base>:src/bts/simulate/mdp.py`).
2. **The decision path:** for every 2027 date from 03-24 to 09-27, every streak 0 to 57, both savers and the same per-bin probabilities, `resolve_policy_decision` on the commit equals the base's in every field except `guard`, unless the state is in the CSV (then `action`, `policy_action`, `pick_bar` and `guard` differ as the CSV says: `pick_bar` becomes None because the action is no longer a skip).
3. **`effective_pick_bar`** is equal to the base over the same grid (the first playing bin is never a guarded one). After G no reach-57 table decision is a skip at or above its bar: assert it over the grid.
4. The tail regime, the heuristic path (no artifact), a corrupt base with a valid tail, `allow_double=False`, and no second pick: equal to the base over the existing fixtures; no guard.
5. A table with tied or missing boundaries, a table narrower than the bins, a `q` clamp: the guarded function behaves as `lookup_action` does for the table's own action (same exceptions, same clamps) and never raises where it does not.
6. Pick files and decision records written by the base load on the commit (no `guard`, no `policy_guard`): read as None. Files written by the commit load under the base (`PolicyDecision` is not reconstructed from disk; `parse_decision` ignores the key): show it.
7. The skip-policy shadow, the boundary census and `is_reach57_mdp_skip` read only skips: unchanged for any record that is still a skip.

## Standing rows
An interrupt mid-step (the lookup is pure; an interrupt between the decision and the record leaves what the base leaves); a malformed or non-regular artifact (the loader is untouched: same errors, same forced rule in the tail regime); midnight ET and the year boundary (the day count is the base's; December 31 and January 1 have no reach-57 play); a failed or unconfirmed send and a restart mid-step (the guard travels on the pick file's `policy_decision`, so a cached-fallback or restart commit records the same `policy_guard` as a fresh one: test both).

## The season walk
Re-run `scripts/audit/season_walk/walk.py` end to end on the commit and compare record by record with `results/walk-bc143131.json.gz` (4,599 runs, 841 flagged skip messages). Expected: every run whose state (streak, days left, saver) and candidate bin are a row of the CSV now plays that row's `guarded_action` (the walk's 0.96 candidate is bin 4: a single) and so has no skip message; no other run's action, bar or day count changes; the runs that skip because EVERY bin skips (streak 56 with 74 or more days left, and the like) still skip and are still flagged, because the skip notice's wording is `main`'s (change set E is stopped). The closing test split the 841 as 229 of the first kind and 612 of the second: state the three counts found (now playing, still flagged, any other change) and save the run as `results/walk-g.txt` and `results/walk-g.json.gz`. If the counts are not 229, 612 and 0, STOP and report before committing.

## Tests (first, RED on the base)
`tests/test_wr_g_guard.py`: the CSV-driven must-change test; rows 1 to 7; the journal line; the record round trips. The CSV is regenerated in a test from the artifact and compared byte for byte with the committed file. Existing assertions that may change: only those that pin a skip in one of the 1,511 states (list each with its state; if there is none, say so). Anything else: STOP and ask.

## Known and not fixed
- The flat top of the model's probability scale (the cause) stays for the longest-streak policy work (D7).
- States where every bin skips are unchanged.
- Offline evaluators of the artifact do not apply the guard; a comparison of "production policy" against a table must apply `lookup_action_guarded` itself.
- The effect on the chance of reaching 57 is not measured (the table's own profiles are lost).

## Handover
`HANDOVER-WR-G.md` in the worktree root (not committed), with: the commit, per contract section the test that was RED first, the grid sizes actually run, the walk's three counts; then `WRG-<first 8 hex of its sha256>` on its own line and `DONE`.

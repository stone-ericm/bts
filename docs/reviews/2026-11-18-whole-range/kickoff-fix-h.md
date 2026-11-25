# Kickoff: change set H, the longest-streak regime from streak 57 (2026-11-25, version 1)

Rulings R62 and R65 (`FINDINGS.md`); Eric 2026-11-24: "after 56, try to achieve the longest streak possible". Tier A (pick choice). Session `bts-wr-fresh11`. Base: `main` at `71590310`. Branch `fix/wr-h`.

## What is wrong on `main`
At streak 57 or more `bts.simulate.mdp.lookup_action` returns `skip` on every day (`if streak >= 57 or days_remaining <= 0: return "skip"`), and `tail_policy.mdp_objective` sends `streak >= target` to that lookup. A 57 streak therefore never plays again. Eric's objective from there is the longest streak possible.

## The design (R65; the session's facts are in `~/bts-runs/wr-fresh11-h-design-question.md`, copied below where the contract relies on them)
1. **One constant.** `bts.simulate.tail_policy.LONGEST_STREAK_FROM = 57`. The new regime governs a state iff `streak >= LONGEST_STREAK_FROM` and the effective day count `d` (`effective_days(_days_remaining(date), season_length)`, today included) is at least 1. It is selected from STATE ALONE, before the reach-57 table, the tail policy and the heuristic are consulted, in the one place the regime is decided (`mdp_objective`, used by `strategy._regime`). With `d <= 0` nothing changes (today's path: skip). Nothing may depend on the constant being 57 except through the constant (Eric is being asked whether it becomes 56; that answer must change one value).
2. **The rule.** Objective: expected longest streak, uncapped, with one quality bin (the tail policy's own rates `bin_p_hit[0]` = p and `bin_p_both[0]` = pb; shipped values 0.73361111 and 0.55111111) and a miss ending the run at the best so far. The expected further gain with `d` days left is `g(0) = 0`, `g(d) = max(g(d-1), p*(1 + g(d-1)), pb*(2 + g(d-1)))`. The action for `d` is the maximiser with ties resolved **single, then double, then skip** (the tail solver's play-first order; at d = 180 the float values of skip and single are equal on the shipped rates, so the order matters). A new pure function in `tail_policy.py`, `longest_streak_action(days: int, p_hit: float, p_both: float) -> str`, computes it by that recursion (no table, no artifact of its own). With the shipped rates it returns `double` for d = 1, 2, 3 and `single` for every d from 4 to 180, and never `skip`. It does not depend on the streak, the saver, the best streak or the candidate's probability.
3. **Where the rates come from.** The loaded and validated tail policy already in the decision context (`mdp["tail"]`, sha-bound to the base policy as today; the same pairing check `_tail_action` makes). When it is unavailable for ANY reason (no artifacts, base unreadable, sha mismatch, tail invalid, the rule function raising): the action is `single` with `degraded_reason` set (`longest-streak rule unavailable (<why>); forced single`), logged as the tail's forced fallback is. Never `skip`, never the reach-57 table, never the heuristic.
4. **The clamps are unchanged and come after:** `allow_double` false turns `double` into `single`; no executable different-game second pick turns `double` into `single`.
5. **The decision.** `PolicyDecision` in the new regime: `objective = "longest_streak"` (new constant `OBJECTIVE_LONGEST` in `tail_policy.py`), `source = "mdp"`, `policy_action` the rule's action, `action` after the clamps, `streak`, `days_effective`, `best_supplied` and `best_status` as supplied, `effective_best = None`, `tail_sha256` = the tail artifact's sha when its rates were used and `None` when degraded, `degraded_reason` as in item 3, `reason = None`, `pick_bar = None` (there is no skip). `daily_decision.OBJECTIVES` gains `"longest_streak"` (without it `write_decision` returns None and the day has no record). The journal line `Policy: objective=longest_streak …` appears through the existing `_log_policy_decision` (it prints whenever the objective is not `reach57`).
6. **`effective_pick_bar`** in the new regime with `d >= 1` returns `0.0` (every probability plays); it never raises.
7. **Readers of `objective`** (each must be shown to treat `"longest_streak"` as "not reach57", as it treats the tail's value today): `daily_decision.decision_objective` and `is_reach57_mdp_skip` (skip-policy shadow and census estimand: excluded), `health/mdp_policy_alignment.py` (excluded from bin utilisation), `scripts/audit/boundary_shadow_census.py` (excluded), `scheduler._selection_decision_meta` and `_commit_decision_for_pick` (recorded, from the selection and from a saved pick's `policy_decision`), `scheduler.py:1718` (the skip summary's `objective` field: no skip occurs here), `scripts/audit/season_walk/walk.py` (records it), `scripts/audit/bin_collapse_read.py` and `p05_tail_policy_audit.py` (offline tools: not changed; noted). The `tail_policy` health check: unchanged and silent on the shipped artifacts.

## Must change (RED on `71590310`)
| # | State | `main` | H |
|---|---|---|---|
| H1 | streak 57, any saver, any probability from 0.01 to 0.99, every d from 4 to 180, a different-game second pick, doubles allowed | skip | single, objective `longest_streak`, source `mdp`, tail sha recorded |
| H2 | the same with d = 1, 2, 3 | skip | double |
| H3 | H2 with `allow_double` false, or with no different-game second pick | skip | single (policy_action double) |
| H4 | streak 57, tail artifact missing, corrupt, a directory or a named pipe at its path, sha-mismatched, or base policy unreadable | skip (or the heuristic when nothing is on disk) | single with `degraded_reason`; no hang (the existing loader's bounded read) |
| H5 | streak 57 through the real decision call `orchestrator.run_and_pick` with a dated observation 57/57/used on 2027-07-15 and on the finale 2027-09-26 | skip | single; double on 09-24, 09-25 and 09-26; the pick file's `policy_decision` and `decision.json` both say `longest_streak` |
| H6 | `effective_pick_bar(57, date, saver)` in season | None | 0.0 |
| H7 | `write_decision(objective="longest_streak")` | returns None, no file | writes the record; `decision_objective` returns `longest_streak`; `is_reach57_mdp_skip` false |
| H8 | a saved pick with `policy_decision.objective == "longest_streak"` committed by the cached fallback or after a restart (selection None) | no record (rejected objective) | the record carries `longest_streak` |

## Must still happen (the proof that nothing else moves)
- **Every state below the boundary is byte-for-byte untouched:** for every streak 0 to 56, every d from 0 to 200, both savers, a grid of probabilities that includes every bin edge and both sides of it, best in {None, trusted, untrusted} over 0 to 57, with and without a second pick, both `allow_double` values, and with the artifacts present, tail-missing and all-missing: `resolve_policy_decision` and `effective_pick_bar` return exactly what the base returns (compare against `git archive 71590310 src`, confirming `bts.__file__`). Likewise `mdp.lookup_action` itself is NOT changed: its `streak >= 57` skip stays (other callers and the solver tooling use it); the regime is decided before it is reached.
- Streak 57 or more with `d <= 0` (the day after the finale and later, and a date in a year with no configured boundary past its fallback): skip, exactly as the base.
- The tail regime and its stop rule, the forced tail action, the heuristic path below the boundary, the skip notice texts, the saver rule, `chosen_on`, delivery, the cutoff, the fallbacks: unchanged.
- The season walk on the commit: its 4,599 existing runs equal the saved records (action, bar, day counts, objective); the walk gains ONE dated state `57/57/used` (219 more runs) whose in-season dates all play, whose finale and the two days before it are doubles, and whose post-finale dates skip; new result files `results/walk-h.txt` and `.json.gz`.
- The existing tests that pin a skip at streak 57 or more are listed by the contract check BY RUNNING the fast-suite files that touch `strategy`, `tail_policy`, `mdp`, the scheduler's decision path and the walk with the boundary patched to 57 in a scratch copy; each is ruled here before the author starts. The author stops and asks on any other assertion that must change.

## Standing rows
| Row | In this change |
|---|---|
| An interrupt mid-step | the rule is a pure function inside `resolve_policy_decision`; a `KeyboardInterrupt` or `SystemExit` raised inside it propagates exactly as on the base (only `Exception` is turned into the forced single) |
| A malformed or non-regular file | H4: the tail artifact or the base policy as a directory, a named pipe, truncated, oversized, wrong schema: forced single with the reason; no hang, no raise |
| Midnight ET and the year boundary | the day count comes from the slate date as today; 2027-09-26 is d = 1 (double), 2027-09-27 is d = 0 (skip, base path); 2027-12-31 and 2028-01-01 at streak 57: the base path's answer for d = 0 and for an unconfigured year, unchanged |
| A failed or unconfirmed send | not touched; a pick chosen in the new regime goes through the same delivery, receipt and `DELIVERY OUTCOME UNKNOWN` paths |
| A restart mid-step | H8: the saved pick's `policy_decision` round-trips through the pick file and is recorded |
| A rollback to the base code | a `decision.json` with `objective: "longest_streak"` read by the base: `decision_objective` returns `unknown` (excluded from the reach-57 consumers), `parse_decision` accepts the record; show both |

## Known and not fixed (limits; each must be true as written, the contract check says if not)
1. **The state checks still refuse a streak or best above 57** (dated observation checks 5 and 6, the automatic-file refusal of change set 1, the fetch's best check, `set-contest-streak --best-streak`, `picks._valid_chosen_on`). So after a hit at 57 there is no accepted live state and no pick, with the existing contest-state CRITICAL. Raising them is change set H2 (R65), right after H. Until H2, H acts on exactly one streak value in live play: 57.
2. With one bin the rule cannot wait for a strong day; it plays every day.
3. A best above the streak (possible only after H2) is ignored: the rule assumes the current run is the best. A miss is valued as ending at the best so far (the small chance of a second run past it later in the season is ignored).
4. Days are calendar days through the finale, as everywhere in the policy; off-days are not modelled, so "3 or fewer days left" can be two game days.
5. The rates are the tail policy's late-season rates applied all season.
6. The offline tools `bin_collapse_read.py` and `p05_tail_policy_audit.py` do not know the new objective.

## The author's work (tests first; ONE commit)
Files: `src/bts/simulate/tail_policy.py`, `src/bts/strategy.py`, `src/bts/daily_decision.py`, `scripts/audit/season_walk/walk.py` and its results and tests, a new `tests/test_wr_h_longest.py`, and only those existing assertions the contract check lists. `src/bts/simulate/mdp.py`, the artifacts under `data/models/`, the scheduler, the state checks and every operator text are NOT changed. Handover `HANDOVER-WR-H.md` in the worktree root (not committed): the commit; per row H1 to H8 the test that was RED first; the below-boundary grid's size and result; the walk result; then `WRH-<first 8 hex of its sha256>` on its own line and `DONE`.

## Review
Gate on the committed tree; a fresh Codex xhigh review and a Carbon review of the same commit, at most two rounds; the reviewers' probes adopted as tests; merge to `main` with `--no-ff` on a Codex SIGN.

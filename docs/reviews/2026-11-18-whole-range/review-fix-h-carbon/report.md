# Review: whole-range change set H (`813935bc` on `0a2bec10`)

## Verdict
SIGN

## Blocking findings
None.

## Edits
None.

## Notes
- **N1 (test gap on positive base horizons):** Every positive-horizon longest-streak check in `tests/test_wr_h_longest.py` uses the shipped `season_length = 180` (`== LONGEST_STREAK_HORIZON`), while `base-negative` uses `-1`. A scratch change in `src/bts/strategy.py::_regime` that clamps `OBJECTIVE_LONGEST` days to `mdp["season_length"]` whenever positive (`use_positive_base_horizon` in `probe_7_mutations.py`) passes all 10 focused test files. Production code is right; `probe_1c_rule_and_regime.py` covers this with a SHA-paired `season_length = 2` base at 10 raw days (`days_effective = 10`, `single`, vs `double` if clamped to `2`).
- **N2 (constant-patch edges outside H's scope):** In `probe_1c_rule_and_regime.py`, patching `LONGEST_STREAK_FROM` to `56` or `58` moves every H path cleanly. At `56`, untouched `tests/simulate/test_tail_policy.py::TestObjectiveSwitch::test_reachability_boundary[56-1-reach57]` (outside A10) still expects `reach57`; at `58`, CLI state installation refuses `58` under the existing caps owned by H2 (limit 1).
- **N3 (documented limits verified):** `probe_4c_readers_and_standing_rows.py` replayed limits 1, 2, 6, 7, 8, 9, 10, and 11 against the real code (refused `58/58` fetch leaving `57/57` active; R33 automatic reuse from `57/57` to `0/57`; final-fallback `.70` heuristic skip refresh on `Batter 9` sending cached `Batter 0` while recording the refresh's `reach57`/`heuristic` metadata; unwritten FIFOs blocking `_load_mdp` while `effective_pick_bar` at 57 returns `0.0` without opening them; `0.0%` bar text when no game is eligible). Each limit in the contract and Amendment 1 is stated truthfully.

## 1. The rule and the regime
Ran `probe_1c_rule_and_regime.py` (`.carbon-review/wr-h/probe_1_rule_and_regime.log`).
- Checked `tail_policy.longest_streak_action` against an independent binary64 calculation across days `1..400` (plus `<= 0` returning `skip`) and 237 valid `(p_hit, p_both)` pairs (94,800 calls), plus an exact rational calculation on days `1..60` for `2641/3600` and `1984/3600`. Shipped rates (`0.7336111111111111`, `0.5511111111111111`) return `double` on days `1..3` and `single` on `4..400`, with exact `single`/`skip` binary64 ties on every day `116..400` resolving to `single`. Ties at `(0, 0)`, `(p, 0)`, `(0.5, 0.25)`, `(1.0, 0.5)` and `nextafter` neighbors all resolved in `single, double, skip` order; no positive day returned `skip`.
- Checked `mdp_objective`, `strategy._regime` (182 cases) and `resolve_policy_decision` / `decide_action` (5,376 cases) across streaks `{-100, -1, False, True, 0, 1, 28, 55, 56, 57, 58, 100, 10**12}`, raw days `{0, 1, 2, 3, 4, 28, 29, 90, 115, 116, 180, 181, 269, 271}`, negative `days_remaining`, a SHA-paired `season_length = 2` base, both clamps (`allow_double`, `has_diff_game`), and all 14 `PolicyDecision` fields (`streak` normalized to `int`, `effective_best = guard = reason = pick_bar = None`).
- Checked diff lines in `src/` and `walk.py`: only `LONGEST_STREAK_FROM = 57` introduces the literal `57`. Patched `LONGEST_STREAK_FROM` to `56` and `58` in scratch copies of `src/`; `mdp_objective`, `_regime`, `resolve_policy_decision`, `effective_pick_bar`, `select_pick`, and `walk.check` all followed the single constant (see N2).

## 2. Rows H1 to H9 and A3's H4 list
Ran `probe_2_rows_h1_h9_and_a3.py` (`.carbon-review/wr-h/probe_2_rows_h1_h9_and_a3.log`).
- Rebuilt H1–H3 across all raw days `1..271`, streaks `(57, 58, 60, 100)`, both savers, four probabilities, and all four clamp combinations (34,688 checks): all matched expected `action`, `policy_action`, `days_effective`, `longest_streak`, and `tail_sha256`.
- Ran 17 H4/A3 failure modes through real `select_pick`, `attach_provenance`, and `save_pick` on `2027-09-26` (`d=1`) and `2027-07-15` (`d=74`) at `p=0.50` and `p=0.99` (68 runs): tail missing, corrupt, truncated, directory, oversize, SHA-mismatched, valid two-bin, `bin_counts=[null]` (`TypeError` in raw loader), missing key; base missing, corrupt, directory, oversize, SHA-paired `season_length = -1` and `0`; nothing on disk; rule raising `RuntimeError`. Every case returned `single` (`source="mdp"`, `tail_sha256=None`, logged `longest-streak rule unavailable (<why>); forced single`), never `skip`, `double`, `heuristic`, or an exception. `KeyboardInterrupt` and `SystemExit` propagated through `_load_mdp`, `longest_streak_action`, and `effective_pick_bar`.
- Rebuilt H5 on `2027-07-15` (`single`) and `2027-09-24..26` (`double`) for both dated and switched automatic `57/57/used` through `orchestrator.run_and_pick` and separately through `scheduler.run_single_check` -> `_deliver_and_lock_pick` (captured DM) -> `decision.json` -> post-restart `run_single_check` short-circuit. Checked H6 (`0.0` with shipped, missing, raising loader, and an unwritten FIFO under a 0.25s `SIGALRM` without opening it; `None` below boundary on loader failure), H7, H8 (empty-cascade fallback, Warmup fallback, attempted-restart, delivered-restart), and H9/A10's exact test diff.

## 3. Nothing else moves (A6)
Ran `probe_3b_below_boundary_parity.py` (`.carbon-review/wr-h/probe_3_below_boundary_parity.log`) against `git archive 0a2bec10 src` (`bts.__file__` verified for archived base vs worktree `src`).
- Confirmed `git diff 0a2bec10 813935bc -- src/bts/simulate/mdp.py` is empty.
- Compared 700,790 contexts across streaks `0..56`, raw days `0..271` (effective `0..180`), both savers, shipped bin edges + `nextafter`, all 1,511 G guard cells across 4 clamp modes (6,044 checks), heuristic `0.80` and partner-product cutoffs (`0.55`, `0.60`, `0.65` + `nextafter` at `p1 < 1.0`), 2,793 tail stop/play cells across 7 artifact modes, 12 artifact/horizon modes (`90`, `15`, `0`, `-1`, two-bin tail, empty bounds), configured/fallback years (`2025`–`2028`), streaks `57..10**9` at zero raw days, 75 direct `effective_pick_bar` edge calls, and 6 live `select_pick`/save/commit cases. Both processes produced identical SHA-256 `75b4a36501c951a5c6f877eac571cb21b29b5f6ab6a4fba226a057654f889035`; `effective_pick_bar`'s `except Exception` changed no non-raising base return value.

## 4. Readers of the objective (item 7, A5)
Ran `probe_4c_readers_and_standing_rows.py` (`.carbon-review/wr-h/probe_4_readers_and_standing_rows.log`).
- Verified 11 readers on real `longest_streak` single (`2027-07-15`) and double (`2027-09-26`) picks/commits and a synthetic `longest_streak` skip: `decision_objective` (`"longest_streak"`); `is_reach57_mdp_skip` (`False`); `skip_policy_shadow` (`record_skip_from_decision`, `record_pending_skips`, `prune_superseded`, `write_status`: excluded); `mdp_policy_alignment` (counts uncommitted preview per limit 8, excludes finalized record with `n=0`); `boundary_shadow_census` (`primary.n=0`, `excluded_tail_objective`, `diffs={}`, `s2_flat_season.n=2`); scheduler commit/classification writers; `check-pick-entered` + `EntryReceipt` (both double slots confirmed, valid receipt); `day_outcome.gather` (`entered`, `ok=True`, single renders `"Batter 0"` despite W8 runner-up in `double_down`, double renders `"Batter 0 + Batter 1"`); `check-results` + `update_streak` (advances model streak `57 -> 58 -> 60`, replay clean); `post_failure` and `web.py` (`_decision_action`, `load_all_picks`); frozen `season_ledger` (`objective="unknown"`, `objective_raw="longest_streak"`, `pick_policy_objective="longest_streak"`). None raised, mis-scored, or counted `longest_streak` as `reach57`.

## 5. Standing rows
Ran 7 ordinary sequences in `probe_4c_readers_and_standing_rows.py` and `probe_2_rows_h1_h9_and_a3.py`:
- Interrupts propagate (`KeyboardInterrupt`, `SystemExit`); ordinary exceptions force `single`; unwritten FIFOs at either policy path block `_load_mdp` until `SIGALRM` (limit 7).
- Restart between selection and commit (empty-cascade fallback, Warmup fallback, mid-send crash classification, post-DM classification): `policy_decision` round-trips through `{date}.json` into `decision.json`.
- Failed DM (`OSError`) clears `delivery_attempted` with no commit and succeeds on retry; mid-send crash leaves `delivery_attempted=True`, makes zero resend calls, alerts `DELIVERY OUTCOME UNKNOWN`, and commits `locked_unconfirmed` with `longest_streak`.
- `2027-09-26` (`d=1`) doubles; `2027-09-27` (`d=0`) skips (`reach57`) with base and uses the `0.80` heuristic without base; `2027-12-31` skips and accepts the 2027 switch at `23:59 ET`; `2028-01-01 00:01 ET` refuses the 2027 switch until 2028 state is installed, then gives `271` raw / `180` effective days, `single`, and the unconfigured-2028 warning.
- Archived `0a2bec10` parses a finalized H `decision.json` (`decision_objective == "unknown"`, `scoreable == True`) and rejects writing a new record for an uncommitted H saved pick (limit 9).

## 6. The season walk
Ran `.venv/bin/python -m scripts.audit.season_walk.walk --out .carbon-review/wr-h/walk-run.json` (17s, `runs=4818 problems=612`, exit 1) and `probe_6_walk_compare_and_faults.py` (`.carbon-review/wr-h/probe_6_walk_compare_and_faults.log`).
- Fresh output matched `results/walk-h.json.gz` identically; all 4,599 original records, boundary checks, and 612 warnings (`50`: 141, `55`: 157, `56`: 314) matched `results/walk-g.json.gz`.
- The 219 added `dated/57/57/used/0.96` rows yielded 207 singles (23 preseason + 184 in-season), doubles on `2027-09-24..26`, 9 post-finale `reach57` skips, valid `chosen_on` and slot counts on all 210 plays, and 0 new findings.
- Planted 11 faults (in-season and preseason skips, singles on `2027-09-24..26` with both `longest_streak` and `reach57` objectives, bad `chosen_on`, missing double leg): `walk.check` flagged all 11.

## 7. The tests
Ran all 10 listed test files individually under `nice -n 19 /usr/bin/timeout 110`: all passed (`54, 38, 44, 48, 13, 9, 3, 9, 24, 19` = 261 passed). Confirmed `test_tail_policy.py` and `test_wr_g_guard.py` changed only as A10 permits (`probe_2_rows_h1_h9_and_a3.py`). Ran 17 scratch changes in `probe_7_mutations.py` (`.carbon-review/wr-h/probe_7_mutations.log`): 16 were caught immediately (both tie-order flips, swapped operands, day off-by-one, dropped one-bin/positive-horizon/enum checks, unclamped base horizon, skip or non-null sha on degradation, post-load regime checks in `select_pick` and `effective_pick_bar`, `None` bar, `>` boundary, `>= 0` days, `except BaseException`); 1 (`use_positive_base_horizon`) survived (see N1).

## 8. Anything else in the diff
Nothing else in `813935bc` is wrong; tracked file list matches Amendment 1's author file list with no stray changes.

ran: `probe_1c_rule_and_regime.py` (94,800 rule + 5,376 decision + 56/58 constant-patch checks), `probe_2_rows_h1_h9_and_a3.py` (H1–H9 + 17 H4 modes + interrupts), `probe_3b_below_boundary_parity.py` (700,790 archived-`0a2bec10` parity checks), `probe_4c_readers_and_standing_rows.py` (11 readers + 7 standing/limit sequences), full `scripts.audit.season_walk.walk` + `probe_6_walk_compare_and_faults.py`, all 10 listed pytest files individually (261 tests), and `probe_7_mutations.py` (17 scratch changes), all offline with synthetic state and pinned ET clocks; did not run: the full fast or slow test suite, `~/bts-runs/yield_run.sh`, live network calls, or H2's >57 cap changes.

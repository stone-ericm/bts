# SRA4 handover

- S1 v3: `contest_state._contest_season_state` implements the seven rows. `load_decision_streak_state` uses current-year source rounds regardless of recording year; resets row 2 to zero/contest/season_start; retains row 3 as stale; rejects rows 4–6 when required or uses the season-gated model fallback. Unexpired manual overrides bypass the season classification. A3's malformed-pick guards are untouched.
- S7: `_policy_saver_available` reads only now_year and implements reserve semantics for active/used/not_earned/uninitialized, in both contest and model branches. `health.contest_state.check` and `web._saver_season` also use now_year. SaverState.is_available and season_for are untouched.
- Fetch: `cli.fetch_contest_streak` binds saver season to the captured fetch timestamp's New York year. Below-10 initialization is preserved; >=10 promotion requires a source_date from that year. CLI fixtures cover old/current/missing round dates and cold initialization.
- S2 v2: health uses the same season classification; row 4 gets a specific local/profile mismatch CRITICAL, rows 5–6 retain fetch alerts, row 2 is exempt, row 3 tolerates up to two settled current-year picks. Other checks remain.
- Best trust: restored loader-level coverage in `test_contest_state_best_trust.py`: a current recording with last-season source_date and best=18 returns season_start/0 with best_status=untrusted.
- New `tests/test_season_rollover_v3.py`: 129 cases. One parameterized test per table, auto/expired manual and both required modes for S1; unexpired overrides beside rows 2/4/5. The real tracked policy through resolve_policy_decision chooses skip with false and single with not_earned reserve=true at streak 8, day 180, lowest bin. Superseded earlier table expectations and two CLI saver assertions updated; touched CLI decision clocks pinned.

RED before source edits: **39 failed, 90 passed** in the new file: S1 table 16, S2 table 10, saver/policy/caller/fetch cases 13. Baseline wider run: 524 passed, 1 localhost-bind PermissionError in the sandbox.
Requested mutations, each pytest exit 1 and restored byte-for-byte with SHA-256 equality:
- Swap row 2/3 outcomes: `test_s1_v3_table` — 20 failed, 44 passed.
- Return saver.is_available: `test_s7_policy_reserve_table` plus tracked-policy end-to-end case — 5 failed, 12 passed.
- Remove two-pick tolerance: `test_s2_v2_table` — 4 failed, 28 passed.

Final focused command: **644 passed, 1 deselected**; git diff --check clean.
`TZ=America/New_York .venv/bin/python -m pytest tests/test_season_rollover_v3.py tests/test_season_rollover_state.py tests/test_season_rollover_data.py tests/test_contest_state.py tests/test_contest_state_best_trust.py tests/test_contest_state_corrupt_precedence.py tests/health/test_contest_state.py tests/data/test_pull.py tests/data/test_build.py tests/test_data_build_empty.py tests/test_decision_saver_fallback.py tests/test_saver_state.py tests/test_tail_policy_r3_fixes.py tests/test_strategy.py tests/test_web_saver.py tests/test_cli_integration.py -q -p no:cacheprovider --tb=short --deselect=tests/test_web_saver.py::test_do_post_saver_transition_socket_wiring`
The deselected existing test opens localhost sockets (sandbox PermissionError in baseline); new dashboard cases use pure helpers. No full gate or commit/push. S3/S4/S5, scheduler/strategy/predict, and saver_state.py are unchanged against 8479a96.

All specified rows matched. Residual ambiguity: row 3 with a current fetch, no source_date, one local settled pick, and file streak 9 deliberately uses 9. Those fields alone cannot prove that 9 belongs to the current season; implemented exactly as required. The existing uninitialized-saver WARN still says the policy is conservative even before any local settlement, when S7 now supplies true; the existing check/message was preserved per the brief.

`git diff 8479a96 --stat` (tracked files):
```text
 src/bts/cli.py                         | 10 +++--
 src/bts/contest_state.py               | 73 +++++++++++++++++++++++++++-------
 src/bts/health/contest_state.py        | 20 +++++++---
 src/bts/web.py                         |  4 +-
 tests/test_cli_integration.py          | 31 +++++++++------
 tests/test_contest_state.py            |  2 +-
 tests/test_contest_state_best_trust.py |  7 ++--
 tests/test_decision_saver_fallback.py  | 13 +++---
 tests/test_season_rollover_state.py    | 34 +++++++---------
 9 files changed, 126 insertions(+), 68 deletions(-)
```
Also collect the untracked new file `tests/test_season_rollover_v3.py` (129 cases); it is excluded from the tracked diff stat above.

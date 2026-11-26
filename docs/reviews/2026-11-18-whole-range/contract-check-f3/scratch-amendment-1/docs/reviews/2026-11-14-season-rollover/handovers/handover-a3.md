# SRA3 handover

Changed only `src/bts/contest_state.py` and `tests/test_season_rollover_state.py` relative to 66f3992. `_resolved_pick_dates` skips non-object JSON and non-string results before set membership; `_has_unconfirmed_miss` skips non-object JSON. Well-formed string-result behavior is unchanged. No commit/push.

RED before implementation: 55 failed, 10 passed (65 new cases): 48 loader failures across six bad shapes × two seasons × two observation modes × two required-state modes; 4 miss-helper failures; 2 unhashable-result failures; 1 CLI failure. Baseline: 330 passed.
Missing contest state with require_contest_state=True still raises the base ContestStateError (required but missing); other no-observation cases return zero/model_only, and first-fetch observations return zero/stale. Only-current-year bad files cannot enable the saved model streak. Fixture files remain untouched.
CLI regression: `test_s1_cli_preview_ignores_bad_historical_pick` uses CliRunner through preview and the real decision loader with pinned now. Prediction/strategy are fixture mocks; schedule access is mocked; no training/network occurs. It reaches selection with streak 0 after an old [] file.

Mutations (each pytest exit 1; source restored byte-for-byte with SHA-256 equality):
- Remove `_resolved_pick_dates` object check: `test_s1_bad_pick_shape_does_not_break_decision` fails 32 cases and the CLI regression fails; 33 failed, 16 passed.
- Remove `_has_unconfirmed_miss` object check: `test_s1_bad_pick_shape_is_not_unconfirmed_miss` fails 4 cases; 4 failed, 2 passed.
- Remove result string check: `test_s1_nonstring_pick_result_is_not_settled` fails list/dict results; 2 failed, 4 passed.

Same focused command, final result **395 passed**:
`TZ=America/New_York .venv/bin/python -m pytest tests/test_season_rollover_state.py tests/test_season_rollover_data.py tests/test_contest_state.py tests/test_contest_state_best_trust.py tests/test_contest_state_corrupt_precedence.py tests/health/test_contest_state.py tests/data/test_pull.py tests/data/test_build.py tests/test_data_build_empty.py tests/test_decision_saver_fallback.py tests/test_saver_state.py tests/test_tail_policy_r3_fixes.py -q -p no:cacheprovider --tb=short`
`git diff --check`: clean. Other S1–S5 behavior untouched; no full suite run.

`git diff 66f3992 --stat`:
```text
 src/bts/contest_state.py            |  7 ++-
 tests/test_season_rollover_state.py | 91 ++++++++++++++++++++++++++++++++++++-
 2 files changed, 95 insertions(+), 3 deletions(-)
```

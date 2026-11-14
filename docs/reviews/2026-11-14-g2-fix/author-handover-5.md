# Cold-start regression tests

Branch `cold-start-tests`, created from `main` as requested. Tests only; no commit. Production source is unchanged after the temporary mutation was restored.

- Added `tests/test_cold_start_integration.py`, with three ported scenarios and four tests. No training, features, lookups, selection, builder, persistence or delivery function is stubbed.
- `test_cold_pipeline_training_selection_delivery`: 64 synthetic historical PAs, spring-only current-season cache, real refresh/features/all thirteen model fits; pre-lineup predictions are empty, posted lineups yield a real streak-0 MDP single and real chokepoint delivery, scoreable delivered decision and exactly one fake DM. No 2027 parquet or refresh marker is written.
- `test_next_day_prior_lineup_projection`: real slot parsing projects the two returning teams; two additional teams with no prior regular game produce no rows.
- One shared real `run_day` fixture backs `test_opening_day_late_lineup_checks_today` and strict-xfail `test_opening_day_late_lineup_gets_a_last_attempt`, with the exact requested S6 reason. The passing control pins 16:10/17:10/18:10 checks and no later feed request; the desired test requires a delivered decision.
- Added all 33 receipt-precedence permutations to `tests/test_day_outcome_entry_sequence.py`: 1 absence, 2 late/absence, 6 batter/late/absence and 24 full/batter/late/absence orders. That file had no permutation coverage; other classifier tests covered individual states and some pairs, not the full ordering matrix.
- Tests use temporary directories and repository policy artifacts, with fake HTTP, clocks and DM transport. Scratch-artifact writes and the builder differential were not ported. Deterministic provenance prevents indirect Git calls. Real LightGBM fits use one CPU via `LOKY_MAX_CPU_COUNT=1`; no slow marker added.

## Runtime and checks
- Requested sleep probe: `0.20801170798949897` seconds for `sleep(0.01)` (throttled host).
- Initial exact-options new-file run: `3 passed, 1 xfailed in 7.48s`; measured wall time **9.19 seconds**.
- Final new-file run, after fixture identity cleanup: `3 passed, 1 xfailed in 8.29s`; measured wall time **9.66 seconds**, within the 20-second limit (`cold-5-final.log`). The three scenarios share the late-day run between its two assertions.
- New-file command: `TZ=America/New_York .venv/bin/python -m pytest tests/test_cold_start_integration.py -q -p no:cacheprovider` (final diagnostic run added `-rx`; `/usr/bin/time -p` measured wall time).
- Focused command: `TZ=America/New_York .venv/bin/python -m pytest tests/test_day_outcome_entry_sequence.py tests/test_day_outcome.py tests/test_refresh_memo.py -q -p no:cacheprovider` → `101 passed in 5.92s` (`focused-5.log`).
- None-guard mutation: removed only `_refresh_season_data`'s None branch; `test_cold_pipeline_training_selection_delivery` failed at `len(None)`: `1 failed in 5.74s`. Source restored byte-for-byte; `git diff --quiet -- src` passes. Logs: `mutation-5.log`, `mutant-5-none-guard.log`.
- S6 diagnostic: running only the desired test with `--runxfail` produced `1 failed in 7.06s`, specifically `assert decision is not None` with `S6: no decision was saved after the 18:10 check`; no setup error (`s6-red-5.log`). Normal strict-xfail execution passes as expected.
- `git diff --check` passes. No requested work remains unfinished; S6 intentionally remains an expected failure rather than a production change.

## git status --short
```text
 M tests/test_day_outcome_entry_sequence.py
?? tests/test_cold_start_integration.py
```

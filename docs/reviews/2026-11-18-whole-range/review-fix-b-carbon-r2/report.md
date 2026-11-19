# Whole-range fix set B — Round 2 review (`fe593f29`)

## Verdict

**SIGN** · `fe593f29` reviewed against `2dce391c` (round 1) and `9e0ec28a` (`main` base). Both round-1 BLOCKs are resolved and the full B1–B6 diff holds under adversarial probing.

## Blocking findings

None.

## Notes

- **Mutant survivor in status-fallback tests (adopted test provided).** Production `load_decision_streak_state(..., omit_selected_dated=True)` correctly calls `load_saver_state` (`src/bts/contest_state.py:606`), so there is no production defect. However, mutating `omit_selected_dated=True` to skip `load_saver_state` (`2a`) survives `tests/test_wr_b_*.py` and `cs4b1`/`cs4b2`/`contest_fetch` (0 failures), and stubbing `load_saver_state` across `verdict()` (`2b`) survives `tests/test_wr_b_*.py`: in `tests/test_wr_b_edges.py:214–237`, `stop_at_target` fires on the first `Path.read_text` of `saver_state.json` inside the independent `saver:` diagnostic (`src/bts/season_contract.py:230`) before `verdict()` runs (similarly, `streak.json` is hit first in `verdict()` before `verdict(omit_selected_dated=True)`), and `automatic path alone:` does not print `saver_flag_state`. `.codex-review/wr-b-r2/adopt/test_wr_b_r2_review_adopt.py` (11 passed on HEAD) kills both `2a` and `2b` (3 failures each) via read-count checks and uncaught `ValueError`/`RecursionError` payloads (`b"9"*5000`, `b"["*12000+b"]"*12000`) behind an active dated observation.
- **`switch` parameter on `load_decision_streak_state`.** On static files `live_switch` equals `switch_state(picks, now)` identically, so static outcomes would not change without `switch`. Retaining `switch` is justified to preserve `status`'s single-open snapshot invariant (`tests/test_season_contract_cs4b1_edges.py:233`, `tests/test_wr_b_edges.py:276`) so `switch:`, `identifier matches switch:`, `live decision now:`, and `automatic path alone:` cannot tear if `season_contract.json` changes mid-command.
- **Handover wording on future `Z`-separator stamps.** `handover-fix-b-r2.md:64` names and justifies `Z` date/time separators (`2027-07-15Z15:00:00Z`), without explicitly noting that a *future* `Z`-separator stamp (e.g. `2027-07-15Z23:00:00Z` at `16:00Z`) suppresses failure DMs for >6 hours where the base's `.replace("Z", "+00:00")` errored and sent immediately.
- **Python 3.12 JSON recursion depth in existing tests.** `"["*3000+"]"*3000` (`tests/test_season_contract_cs4b1.py:160`) parses as a `list` under CPython 3.12's 10,000 C-recursion limit rather than raising `RecursionError` (`12000` reaches `RecursionError`).

## 1. Pick-path loader against the base

- `git diff 9e0ec28a HEAD -- src/bts/contest_state.py src/bts/saver_state.py` has zero diff in `saver_state.py` and adds only keyword-only `omit_selected_dated: bool = False` (`load_contest_streak_state`, `load_decision_streak_state`) and `switch=None` (`load_decision_streak_state` -> `live_automatic_refusal`). `util.read_utf8` and every `read_text` parameter from `2dce391c` are removed.
- `loader_probe_r2.py` ran **2,112** states on `9e0ec28a` and `fe593f29` with defaults across four ET instants (July 15 midnight and Dec 31 New Year: empty, corrupt/huge/deep/directory fields, cross-product with pick files, `PermissionError` at 6 paths, bounded FIFO blocking at 6 paths). Returned dataclasses, exception types/messages/args/causes, and masked health alerts produced byte-identical pickles (`b80ae6e5…`).
- AST audit of all 37 target calls in `src/` (`loader-summary.json`): only `src/bts/season_contract.py:253` (`inspection_lines`, called solely by `season_contract status`) passes `switch` or `omit_selected_dated`; only `src/bts/cli.py:1725` (`_read_contest_fetch_status`) calls `util.read_regular_file_text`.

## 2. Status verdict equals the live loader

- `status_probe_r2.py` ran **27** ordinary/adversarial regular-file states (four dated/automatic combinations, winning/expired legacy overrides, corrupt manual/auto/switch, wrong `activation_id`, missing/naive/stale/future/year-9999 `recorded_at`, absent/other-season/`9`×5000 `saver_state.json`, settlement gap, `--season` for another year, both sides of Dec 31 midnight ET). In every case `automatic path alone:` equals `load_decision_streak_state(require_contest_state=True)` with today's dated observation absent, and `accepted` is never printed when that call raises.
- Across **26** FIFO/directory cases at all 7 paths (unmasked and masked by dated) plus round 1's exact unsettled-fetch + `saver_state.json` FIFO cutover:
  - **FIFO**: **blocks** in both `status` and live read at `season_contract.json`, `contest_streak.manual.json`, `contest_streak.json`, `saver_state.json`, `streak.json`, and scanned `YYYY-MM-DD.json` (round 1's false `accepted` on `saver`, `model`, and `pick` FIFOs is gone); at `contest_ledger.jsonl` (not opened by live), `status` **blocks** at its ledger diagnostic before output (base behaviour).
  - **Directory**: **raises** `ContestStateError` in live read and prints `refused` in `status` at `season_contract.json` (exit 1), `contest_streak.manual.json`, and `contest_streak.json`; **falls back** and accepts in both at `saver_state.json` (`uninitialized`), `streak.json` (`model_state_error`), and `YYYY-MM-DD.json` (skipped); at `contest_ledger.jsonl`, `status` reports `ledger: unreadable` and prints `accepted`.
- Modes, bytes, and `st_mtime_ns` were unchanged in every run.

## 3. Cooldown stamp

- `cooldown_probe_r2.py` ran 19 stamp shapes × 2 layouts (`last_alert_at`, `last_alert_at_by_category`) = **38** cases on `9e0ec28a`, `2dce391c`, and `fe593f29` across failure, `+8h` retry, success, and post-success failure (`16:00Z` July 15, 2027):
  - Ordinary aware stamps inside (`15:00+00:00`), outside (`09:00+00:00`), and at (`10:00+00:00`) 6h, `9999-12-31T23:59:59-01:00`, `0001-01-01T00:00:00+01:00`, `Z` suffix, and `+0000` without colon: **identical to base** on both lanes (restoring both year-10000 and year-0 aware edges dropped in round 1).
  - Naive (`2027-07-15T15:00:00`), date-only (`2027-07-15`, `2027-07-15Z`), and `42`: base **crashed** on failure (`TypeError` / `AttributeError`, exit 1, no DM, status untouched) and kept them on success; HEAD sends DM, exits 2, writes `now_iso` on failure, and strips them on success (`0`: both send DM/exit 2 on failure; HEAD strips `0` on success). Garbage (`"not-a-timestamp"`): both send DM/exit 2 on failure, except legacy-layout base copied garbage into sibling categories and `last_alert_at` and kept it on success, whereas HEAD strips it on both lanes. All invalid-stamp differences match B5 and the handovers and are right.
- **Suppressing >6h here where base did not:** **Yes — a future aware stamp with uppercase `Z` as the date/time separator** (e.g. `2027-07-15Z23:00:00Z`, verified still suppressed at `+8h`, or `9999-12-31Z23:59:59-01:00`). Neither base nor HEAD caps future stamps (base already suppressed >6h on `2027-07-15 23:00:00+00:00`), and Amendment 2 requires validating with native `datetime.fromisoformat` + offset without UTC date normalization, so honouring it is expected and right.

## 4. Round-1 invariants and adversarial whole-range pass (B1–B6)

- `whole_range_probe_r2.py` verified against `9e0ec28a`: **B1** extracted command exits 0 across spaced, quoted, and relative paths, leaving live `dated` streak 0 accepted and `saver_state.json` bytes/mtime untouched while preserving all other `set-contest-streak` templates and 4c/6b-1 texts; **B2** `locked_unconfirmed` emits verification text at `initial`, `T-30`, and `T-15` (`no_pick` and `mismatch`, single/double, pick-flag disagreement) while stale `decision["date"]` stays on the old path and `delivered`/`private_locked`/`legacy` DMs, markers, receipts, exits, and all-clear match base byte-for-byte; **B3** renders `primary` only on `single` across all 9 pick-carrying classifications without inspecting malformed runner-ups, keeping `double` (`A + B`), non-pick states, exits, and `decision.json` bytes unchanged; **B5** repairs `[]`/`null`/`\xff`/FIFO and diagnoses directory status (exit 2) across switched/unswitched/dry lanes while matching base byte-for-byte on conforming status; **B6** edits only row `B8a` in `docs/ops/2027-season-start.md` with all three gates.

## 5. Tests and mutations

- `git diff 2dce391c HEAD -- tests` removes only the 7 non-blocking FIFO status cases (`test_status_never_blocks_on_special_files`), restores `Path.read_text` spies (`test_season_contract_cs4b1_edges.py:224`, `test_wr_b_edges.py:260,293`), and adds the 19-case ordinary parity matrix, 3-path interruption test, signature check, and 4-case aware cooldown test; nothing else was weakened.
- `mutation_probe_r2.py` on scratch copies: re-threading non-blocking status reads (`1a` round-1 style: **19** failed; `1b` inside `verdict()`: **7** failed), `_reading_time_utc` stamp validation (`3`: **2** failed), and dropping `.utcoffset()` (`4`: **3** failed) were **killed**; ignoring the saver read in status (`2a`/`2b`) **survived** `tests/test_wr_b_*.py` (see Notes and `adopt/test_wr_b_r2_review_adopt.py`).
- Single-process run of `tests/test_wr_b_contract.py`, `tests/test_wr_b_edges.py`, `tests/test_lightgbm_optional.py`, and `tests/test_web_saver.py`: **108 passed, 1 failed** (`test_do_post_saver_transition_socket_wiring` hit sandbox `PermissionError` on loopback socket connect). Targeted controls + 4c/6b-1/6b-2 exact-text suites: **2,238 passed**.

## 6. Remaining diff (`git diff 9e0ec28a HEAD`)

- No other issues found across the 13 changed files. `git diff --check 9e0ec28a HEAD` passed and no tracked file was modified.

## What I ran

- Sandbox denied `~/bts-runs/yield_run.sh` (`Operation not permitted`), so pytest suites ran one at a time via `/usr/bin/nice -n 19 .venv/bin/python -m pytest -p no:cacheprovider …`.
- Probes and evidence retained in `.codex-review/wr-b-r2/`: `loader_probe_r2.py` (`loader-*.pkl`, `loader-summary.json`), `status_probe_r2.py` (`status-{base,r1,head}.json`), `cooldown_probe_r2.py` (`cooldown-{base,r1,head,compare}.json`), `whole_range_probe_r2.py` (`whole-{base,head}.json`), `mutation_probe_r2.py` (`mutant-*.log`, `mutants-summary.json`), `verify_adopt_kills_survivor.py`, and `adopt/test_wr_b_r2_review_adopt.py` (11 passed).

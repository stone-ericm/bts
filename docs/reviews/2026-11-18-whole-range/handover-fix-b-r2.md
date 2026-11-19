# Whole-range fix set B, round 2

Branch `fix/wr-b`. Commit `fe593f29e7da4f7a8d98797cd458213f63ad09fc`, one commit above `2dce391c`.

## Diff stats

Against `2dce391c`:

```text
 src/bts/cli.py                            |  15 ++-
 src/bts/contest_state.py                  |  52 ++++------
 src/bts/saver_state.py                    |   4 +-
 src/bts/season_contract.py                |  18 ++--
 src/bts/util.py                           |   4 -
 tests/test_season_contract_cs4b1_edges.py |   7 +-
 tests/test_wr_b_edges.py                  | 167 +++++++++++++++++++++++++-----
 7 files changed, 187 insertions(+), 80 deletions(-)
```

Against `9e0ec28a`:

```text
 docs/ops/2027-season-start.md             |   2 +-
 src/bts/cli.py                            |  83 ++--
 src/bts/contest_state.py                  |  14 +-
 src/bts/day_outcome.py                    |   9 +-
 src/bts/health/contest_state.py           |   5 +-
 src/bts/season_contract.py                |  46 ++-
 src/bts/util.py                           |  14 +
 tests/test_cli_integration.py             |   2 +-
 tests/test_contest_fetch.py               |   4 +-
 tests/test_season_contract_cs4b1_edges.py |   6 +-
 tests/test_season_contract_cs4b2.py       |   2 +-
 tests/test_wr_b_contract.py               | 180 +++++++++
 tests/test_wr_b_edges.py                  | 610 ++++++++++++++++++++++++++++++
 13 files changed, 932 insertions(+), 45 deletions(-)
```

## B4 status

`src/bts/contest_state.py:368,525`, `src/bts/saver_state.py:45`, `src/bts/season_contract.py:68,184`, and `src/bts/cli.py:2186` restore the ordinary reads for the switch, observations, model streak, pick scan, saver and ledger. The decision sees the same read results and errors as a live call for the state paths it opens. Status diagnostics also use their base readers. A FIFO can block the status command or live loader that opens it. Directories and other read errors follow the base paths for the applicable operation, including existing diagnostics and fallback behavior. The amended contract accepts that behavior.

Status retains `omit_selected_dated` for the fallback evaluated after today's dated observation wins. Status retains `switch` to classify the same switch snapshot for live authority and diagnostics, including when `--season` inspects a different year. This keeps the one-open status contract. The switch snapshot comes from the same `switch_state` reader used by live.

| Function | Before on `2dce391c` | After |
|---|---|---|
| `load_decision_streak_state` | `(picks_dir: Path, *, require_contest_state: bool = False, now: datetime \| None = None, read_text=read_utf8, switch=None, omit_selected_dated: bool = False) -> DecisionStreakState` | `(picks_dir: Path, *, require_contest_state: bool = False, now: datetime \| None = None, switch=None, omit_selected_dated: bool = False) -> DecisionStreakState` |
| `load_saver_state` | `(picks_dir: Path, *, season: int, read_text=None) -> SaverState` | `(picks_dir: Path, *, season: int) -> SaverState` |
| `load_contest_streak_state` | `(picks_dir: Path, *, now: datetime \| None = None, require_dated_auto: bool = False, read_text=read_utf8, omit_selected_dated: bool = False) -> ContestStreakState \| None` | `(picks_dir: Path, *, now: datetime \| None = None, require_dated_auto: bool = False, omit_selected_dated: bool = False) -> ContestStreakState \| None` |
| `switch_state` | `(picks_dir: Path, now: datetime, *, season: int \| None = None, read_text=read_utf8) -> SwitchState` | `(picks_dir: Path, now: datetime, *, season: int \| None = None) -> SwitchState` |

`_parse_state_file`, `_resolved_pick_dates`, `latest_resolved_pick_date`, `resolved_pick_settlement_gap`, `_has_unconfirmed_miss`, `contest_state_is_fresh` and `_load_model_diagnostics` all drop the added keyword-only `read_text=read_utf8` parameter and restore the base bodies. `util.read_utf8` is removed. B5 is the sole caller of `util.read_regular_file_text`.

- `tests/test_wr_b_edges.py:148` replaces the seven nonblocking FIFO expectations with 19 ordinary-file cases from the review probe. The test compares the status line to the required live loader at the same instant, with the selected dated record physically absent for the live read. A malformed, expired or winning legacy record stays on disk. The incomplete-fetch fixture represents the observation already installed with subsequent writes missing.
- `tests/test_wr_b_edges.py:215` adds three interrupted-read cases for the saver, model and scanned-pick paths cited in the review. The status command and live read both reach the ordinary read. Status emits no accepted line and the interrupt leaves the fixture intact.
- `tests/test_wr_b_edges.py:239` adds the signature check against the base after allowing the two status options for the decision loader.
- `tests/test_wr_b_edges.py:254,286` moves read spies from the status-only helper to `Path.read_text` for the New Year and restart tests. The clock-change test checks one switch read and two manual reads, one each for diagnostics and live selection. The restart tests inject `KeyboardInterrupt` and `SystemExit`, check that the fixture is intact, then rerun with ordinary reads.
- `tests/test_season_contract_cs4b1_edges.py:218` restores its `Path.read_text` spy from the `os.open` spy. It still checks one switch read.

## B5 cooldown stamps

`src/bts/cli.py:1719,1761`. Both fetch-status read lanes validate with `datetime.fromisoformat`, the parser the cooldown subtraction uses. Validation requires an offset. It preserves the stored string and avoids converting through a UTC calendar date. The round-one regression stamp `9999-12-31T23:59:59-01:00` suppresses a DM on both the base and this commit. Round one dropped it while trying to normalize to UTC year 10000.

The base rewrote every uppercase `Z` to `+00:00`. Canonical trailing `Z` and explicit numeric offsets have the same cooldown semantics here. An uppercase `Z` used as the one-character date/time separator, followed by a valid offset, is also legal to the native parser. For example, `2027-07-15Z15:00:00Z` and `2027-07-15Z15:00:00-01:00` now retain cooldown because native parsing accepts their separator and finds an offset. The base's global substitution corrupted that separator. A separator-only `Z` without a timezone still leaves a naive timestamp and validation discards it.

- `tests/test_wr_b_edges.py:398` adds four offline command sequences for the two storage layouts and the extreme-offset and unusual-separator stamps. Each sequence verifies the initial successful fetch preserves the seeded stamp, a throttled failure leaves it intact, and a successful fetch followed by another failure still suppresses the DM. Existing malformed-file, interrupted-write, restart and unconfirmed-send controls stay in the focused suite.

## Runs

- Before changing production code, an offline Python check on `2dce391c` showed the extra loader parameters and showed both year-edge stamps discarded from a synthetic status file despite successful aware subtraction. The round-one report documents the original FIFO reproductions.
- After the commit, an offline status-file check retained the extreme offset and both examples with a `Z` separator and dropped the naive separator-only example.
- `git diff --check 9e0ec28a HEAD` passed.
- The focused wrapper ran after 01:58 ET on the committed tree and finished with 2,001 passed and one deselected. It included both whole-range B files, LightGBM optional, web saver, the 4b-1 and 4b-2 command contracts, contest fetch, day outcome and entry sequence. The sandboxed run deselected the one socket-binding web saver test, which is included in the unsandboxed fast gate. See `.codex-review/wr-b/round2-focused.log`.
- The single unsandboxed fast gate on the committed tree finished with 9,326 passed, 37 failed, 16 errors, 25 skipped, 6 deselected and 1 xfailed. Its 53 failed or errored node IDs matched `~/bts-runs/gate-failed.txt` exactly, with zero extra and zero missing. Both whole-range B files, LightGBM optional and all 12 web saver tests passed together. The wrapper exited 1 for the saved failures. See `.codex-review/wr-b/round2-fast.log` and its `.watch` file.

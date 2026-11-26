# Kickoff: change set 2, second attempt (the season tag on `streak.json`; this time the date handling)

Tier A (pick-path behaviour): tests first and RED, then the change, then the gate. A fresh adversarial review of the WHOLE change set follows your handover; write for a reader who assumes you are wrong.

You work in `~/projects/bts-work/cs2b`, branch `sb-cs2b`, cut from `sb-cs2` at `ee9fb870`. Commit there. Do not push. No network, no box, no contest account. Python is `.venv/bin/python` in that worktree (already synced). This machine is shared: run EVERY pytest invocation as `~/bts-runs/yield_run.sh <a log file inside the worktree's .codex-review/cs2b/> env TZ=America/New_York PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider …` and read the log; never use a pattern kill (`pkill`, `killall`). Write any helper script inside `.codex-review/cs2b/`, not in `/tmp`.

## Where this stands
Read `docs/reviews/2026-11-14-season-rollover/build/STOP-CS2.md` first (it is on `main`; a copy is in `.codex-review/cs2b/`). The first attempt is the branch you start from. Three review rounds found it sound except for one thing, which blocked it twice: **how a date string becomes a season.** Everything else on the branch stays as it is. Your change is small. Its difficulty is that the same mistake was made twice, so the rule below is to be implemented exactly and tested against the base's behaviour, not against a parser's.

`d8eb50d0` is "the base": `main` before this change set. `git diff d8eb50d0 HEAD` is the whole change set.

## The full contract (this document replaces `kickoff-cs2.md` and its amendments; rows A to K restate what the branch already does and you must not change, row L is your work)

**Definitions.** The season of a date is its calendar year. A `streak.json` matches season Y when it is a JSON object whose `season` is the integer Y (`type(x) is int`). A readable object with another tag, or none, matches no season and reads as season-fresh: streak 0, saver `True`, applied dates `[]`. A file that cannot be read as a JSON object is CORRUPT: that is an error, never "another season".

| # | Rule | Status |
|---|---|---|
| A | `load_streak`, `load_saver_available`, `load_applied_dates` take keyword-only `season: int \| None = None`. `None`: the base's behaviour byte for byte, including what it raises. An int: season-fresh for a readable non-matching object; the file's values for a matching one; on a corrupt file the streak and saver readers raise exactly what they raise with `None`, and `load_applied_dates` returns `[]`. A `season` that is not an int or `None` is a `TypeError`. Reads never write | on the branch |
| B | `save_streak(..., season=Y)` stamps `"season": Y` and carries the saver and the applied dates only from a matching file. With `season=None` it behaves as at the base and writes back any existing `season` value unchanged. On a corrupt file a seasonal save raises unless it is given an explicit saver (the reconcile replay's rebuild), in which case it replaces the file, as at the base | on the branch |
| C | A dated `update_streak` reads and writes with the season of its date, so a result on a non-matching readable file is applied to season-fresh state; on a corrupt file it raises before writing and the bytes are preserved. The once-per-date rule and the saver rule are unchanged on a matching file | on the branch, except the date handling (row L) |
| D | A dateless `update_streak` never resets and keeps the file's tag | on the branch |
| E | `reconcile_results` saves its replay with `season=replay_day.year`; `_replay_season_streak`'s marker check stays without a season | on the branch |
| F | The reconcile command's summary reads the streak with the ET year of the receipt's clock | on the branch |
| G | `contest_state._load_model_diagnostics` returns `(0, True, None)` for a readable object that does not match the ET year of `now`; a corrupt file is the error change set 1 made it | on the branch |
| H | A tagless file is not rewritten by a read; the first dated write stamps it | on the branch |
| I | `web.load_streak`, `health/streak_validation`, `state/export.py`, `state/verify.py`, `state/regenerate.py`, `data/backup.py` are unchanged | on the branch |
| J | The existing tests changed by the first attempt keep their changes (season-tagged fixtures, the clock pins) | on the branch |
| K | Known limit, by design: a mid-season deploy reads the model streak as 0 until the 02:00 reconcile | unchanged |
| **L** | **How a date becomes a season: see below** | **your work** |

### Row L
**One function decides, in one place, and it never raises.** Add to `src/bts/picks.py`:

`season_of_date(date) -> int | None`: for a `str`, the year from `datetime.date.fromisoformat(date)`, and if that raises, the year from `datetime.datetime.strptime(date, "%Y-%m-%d")`, and if that raises too, `None`. For anything that is not a `str` (a `date` object, `None`, bytes, an int), `None`. No stripping, no other formats, no regular expression.

Then:

| Site | At the base | On the branch (`ee9fb870`) | Required |
|---|---|---|---|
| L1. The four reads in `scheduler.py` that pass a season to `load_streak` (the shadow selection's streak; the three display reads in `run_result_polling`) | no season, no parse: any date value is accepted | `_season_of(date)` using `strptime`: raises on compact (`20270324`) and week (`2027-W12-3`) dates and on everything else `strptime` rejects | `season=season_of_date(date)`. A date whose season cannot be told gives `season=None`, which is the base's read. Delete `_season_of` |
| L2. The CLI's void read in `check-results` (`cli.py`, the `else load_streak(...)` beside `update_streak`) | no season | `date_type.fromisoformat(date).year` | `season=season_of_date(date)` |
| L3. `strategy.select_pick`'s saver fallback | no season | `strptime(date).year` | `season=season_of_date(date)` |
| L4. `update_streak(results, picks_dir, date=D)` with `D` not `None` | any value was accepted and used as the once-per-date marker | a strict regular expression plus `fromisoformat`; anything else raises `ValueError` | `season_of_date(D)`; if it is `None`, raise `ValueError` before any read or write (a scorer must not write without knowing the season). Otherwise proceed with that season, and the marker stored is `D` exactly as given, as at the base. So an unpadded or compact date that the base scored is scored again |

The consequence to prove, and the reason for the rule: **for every date value, each of L1 to L3 either reads the season-scoped value (the date can be read) or does exactly what the base did (it cannot). None of them can fail because of the date.** L4 fails only when the season cannot be determined.

Do not change `_now_et`, the CLI's own age-guard date parsing, or anything else about what dates the commands accept upstream.

## Tests
- RED first: write each test, run it on the branch as it is, record the failure, then make the change.
- `season_of_date`, as a table: `"2027-03-24"` → 2027; `"2027-3-24"`, `"2027-03-4"`, `"2027-3-4"` → 2027; `"20270324"` → 2027; `"2027-W12-3"` → 2027; `" 2027-03-24"`, `"2027-03-24 "`, `"2027-03-24T10:00"`, `"2027/03/24"`, `"2027.03.24"`, `"2027-02-30"`, `"junk"`, `""` → `None`; `datetime.date(2027, 3, 24)`, `None`, `b"2027-03-24"`, `20270324` → `None`. If this interpreter disagrees with a row, STOP and report it; do not adjust the function to the table or the table to the function.
- L1, through the REAL entry points, one parametrised test per path over the whole table above: `scheduler._run_shadow_prediction`, and `run_result_polling` on its all-void, final-peer and early-peer branches (the first attempt's tests `test_row8_shadow_selection_uses_own_date_season` and `test_row6_scheduler_display_reads_scored_season` show how to drive them; extend those, do not write shallower ones). Fixture: `streak.json` tagged 2026 at streak 13. Expected: no exception from the season read for ANY row; a date whose season is 2027 reads 0; a date whose season cannot be told reads 13 (the base's value). Where a path cannot be driven with a non-string date because something upstream of the read rejects it at the base too, say so per row and show the base's behaviour.
- **The comparison with the base, as a recorded run, not a claim.** Put the base's `src` in a scratch copy (`git archive d8eb50d0 src | tar -x -C .codex-review/cs2b/base`, never a checkout over the tracked tree) and write `.codex-review/cs2b/date_matrix.py`: for every row of the table and every path in L1 to L4, run the base package and the branch package and record the outcome (return value, exception type, the pick file's result, `streak.json` bytes before and after). Commit the script's summary table into the handover. Every difference between the two must be one of: (a) a parseable date now reads or writes the season-scoped value, (b) L4 raising `ValueError` for a date whose season cannot be told. Anything else is a defect.
- L4: the dates the base scored and the first attempt rejected (`"2027-3-24"`, `"20270324"`, `"2027-W12-3"`) score again, stamp season 2027, and store the marker as given; a second call with the same string changes nothing; the values with season `None` raise `ValueError` with the file's bytes unchanged and zero reads (the first attempt's tests `test_row3_*` assert the stricter rule: change them to this one and say so).
- The first attempt's tests otherwise stay. Do not delete or loosen an assertion to get green; list every changed test with the reason.
- No test may depend on today's date. Pass or pin every clock.
- Before you start and after you finish, run every test file that mentions `streak.json`, `load_streak`, `save_streak`, `update_streak`, `load_saver_available`, `load_applied_dates`, `reconcile_results`, `select_pick` or `run_result_polling` (find them with `rg`), then the fast suite: `… -m pytest -m "not slow" --ignore=tests/simulate --ignore=tests/model --ignore=tests/experiment --ignore=tests/validate -q -p no:cacheprovider` (about 25 minutes through the wrapper). On this machine 53 tests fail for environment reasons: the list is `~/bts-runs/gate-failed.txt`; your run is clean when the failing set is exactly that list. Take the baseline counts BEFORE you create any new test file (an untracked test file makes one existing test fail). Report the counts before and after.

## Handover
Write `HANDOVER-CS2B.md` in the worktree root (do not commit it): the commits; for L1 to L4 the file and lines and the tests; the date matrix's summary table and every base-versus-branch difference with its class (a) or (b); every changed test and why; every remaining place in `git diff d8eb50d0 HEAD -- src` that parses a date or derives a season, with what it uses; what you ran with the counts; anything you could not match or think is wrong, including in rows A to K. Then print `CS2B-<first 8 hex characters of the sha256 of HANDOVER-CS2B.md>` on its own line and `DONE`.

## Amendment to row L after review round 1 (lead session, 2026-11-16): an exact `str`, nothing else
Row L said `season_of_date` "never raises" and tested for a `str` with `isinstance`. The Codex reviewer (marker `CS2BRC-2f9ff754`, B1 and R1) showed two objects that break that promise: a non-string whose `__class__` property raises (so `isinstance` itself raises), and a `str` subclass whose `__len__` raises `SystemExit` or blocks (the parsers call it). Neither can come from command-line text or JSON; the finding is against the promise as written, and the promise was the lead's.

**Row L is amended:** only an exact built-in string is parsed: `type(date) is str`. Anything else, including a `str` subclass, has season `None`. `type()` consults nothing on the value and the parsers then see a real `str`, so no attribute or method of the caller's object runs inside the helper: it cannot raise, exit or stall. A `str` subclass therefore reads as at the base in L1 to L3 and is refused by a dated `update_streak` with `ValueError` before any I/O (class (b) of the allowed differences). The `ValueError` message shows the value only when it is an exact `str`, and the type's name otherwise.

Cost if wrong: code that passes a `str` subclass as a contest date (none exists in `src` or `scripts`) gets the base's unscoped read, and its scoring call is refused loudly.

## After review round 2 (2026-11-16): the rejection message, and one confirmation round by the owner's decision
Round 2 (Codex, marker `CS2BRC2-a0790b29`) closed the round-1 findings and blocked on B2: the `ValueError` message of a dated `update_streak` named the value's type with `type(date).__name__`, which a metaclass can turn into caller code. **Row L, completed:** between entering a dated `update_streak` and raising for a date with no season, the only operation on the date is `type(date) is str`; the message shows `repr(date)` for an exact `str` and fixed text otherwise. Commit `0a34b7cf`. Put to Eric with three options, he chose the first ("1"): one narrow confirmation by a fresh Codex reviewer of `c9c1e342..0a34b7cf`, the full gate, and merge if both are clean. If the confirmation blocks, the change set goes back to him.

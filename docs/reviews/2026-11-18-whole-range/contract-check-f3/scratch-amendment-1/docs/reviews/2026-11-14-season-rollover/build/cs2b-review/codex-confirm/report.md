# Verdict

**SIGN WITH EDITS** — B2 is closed at 0a34b7cf; no production blocker. Correct the remaining coverage-header overstatement below. Scope: only `git diff c9c1e342 HEAD` and behavior affected by it.

# B2

**Closed.** `r3_probe.py` executes the four original classes from `r2_attack.py` through real `season_of_date` and `update_streak` on real directories: metaclass name lookup, name-string formatting, sentinel write, and 60-second sleep.
All four now raise exactly `ValueError`, with no hook calls, no state-reader/writer calls, no audited file operations, and identical directory/file bytes. The sleep case returns immediately. `r3-attacks.json` retains exact before/after bytes and observations.
Five independent cases also pass: an object and a str subclass with hostile value/metaclass hooks, the hostile class itself as D, an object whose equality/truth/repr/str exit, and an opaque object. Wrapped state functions would execute their real bodies if reached; none is reached.
The scratch c9c1e342 package reproduces RuntimeError, SystemExit, sentinel I/O, and the sleeping child timing out. Only that subprocess is killed by its timeout; no pattern kill.
The only intended production difference is the fixed `<not a str>` rejection message for non-exact strings; parseable dates, dateless updates, and exact-string rejection messages retain their behavior.

# Every expression that touches the date

Exhaustive rejection path, in execution order (`src/bts/picks.py`; D is not None):
1. Line 688: `date is not None` — an identity comparison, never equality/truth/attribute dispatch.
2. Line 689: `season_of_date(date)` — pass/bind the same reference to the helper.
3. Line 526: `type(date) is not str` — builtin type plus identity; no value or metaclass hooks. Non-exact strings return None immediately.
4. Exact strings only, line 529: `date_type.fromisoformat(date).year`; on Exception, line 532: `datetime.strptime(date, "%Y-%m-%d").year`. The year attribute belongs to a builtin parser result, not D.
5. Line 693: `type(date) is str`, then `repr(date)` only for that exact builtin str; otherwise the fixed literal.
6. Line 694 concatenates a constant and the already-derived exact string and constructs/raises builtin ValueError. It performs no further operation on D. The `season is None` check likewise touches only the helper result.
Thus the literal answer to “ANY operation other than type/parsers/repr?” includes the unchanged, inert None identity check and reference passing. There is no additional caller-dispatching operation. Line 695's first state read and line 697's membership test are unreachable on rejection.

For exact builtin strings, repr and the parser argument cannot dispatch methods supplied by D. Both parser calls and their `.year` lookups are inside `except Exception` boundaries. Repr returns an exact str, so message concatenation cannot dispatch caller formatting code.
Real calls cover 1,352 strings, including all ASCII substitutions at every position of `2027-03-24`, Unicode/surrogates/NULs, invalid dates, and three 5,000,000-character inputs. All 1,271 rejected strings raise ValueError before state access and preserve bytes; no input-controlled escape or callback was found (`r3-exact-strings.json`).
Interpreter-wide conditions are excluded: signals/process cancellation, global tracing/import hooks or monkeypatches, and resource collapse can interrupt any Python operation, including constructing the promised ValueError. A real timer signal while parsing exact `"junk"` reproduces escaping KeyboardInterrupt; it is independent of date contents. Parser MemoryError/RecursionError are Exception and caught, but repr/message/exception allocation can itself fail when memory is unavailable. This is not an unconditional guarantee against interpreter failure.

There are **six**, not five, read sites: scheduler.py:1899,2561,2605,2610; cli.py:2438; strategy.py:573. Each passes D to the same helper; only exact strings reach its parsers, and unknown seasons retain the unscoped read. These complete source files and the helper are byte-identical to c9c1e342 (`r3-source-equivalence.json`). This commit changes none of those read expressions or their surrounding restrictions.

# Blocking findings

None.

# Notes

- **Coverage-header correction remains incomplete (test documentation only).** `tests/test_streak_season_cs2b_review.py:15` correctly describes its chmod-000 corruption case, including scorer rejection and complete-replay rebuild. Lines 16–17 still say “The other corrupt payloads are covered” in the two named files. Those files cover `{`, non-UTF-8, `[]`, null, a number, and deep nesting; neither tests an empty file, a JSON string, or a >4300-digit integer. DEEP_JSON/HUGE_INT_JSON in the adopted file remain unused. Smallest edit: explicitly list the corruption cases actually covered by the two cited files, rather than claiming the other payloads collectively. This is a coverage claim, not a production-behavior defect.
- **New regression case is effective.** Current code passes both `type-name-trap` tests. With actual c9c1e342 src extracted into this review directory, the new dated-update case fails at tests/test_streak_season_cs2.py:94 because it catches `RuntimeError('metaclass name hook')` instead of ValueError; the helper case passes. JUnit and full failure output are retained.
- **Adopted race correction is sound.** The actual test trace records two resolver entries (poller and one CLI peer), exactly one dated update in the peer, streak 1→3, then polling's locked peer-result branch at scheduler.py:2605 observing `hit`/3. Polling does not update again. The guard is set before entering the peer, so recursion ends. Every pre-existing assert condition in the entire adopted file is AST-identical; only the CLI assertion's failure diagnostic changes. Existing reply/streak/crash/replay assertions remain intact (`r3-race-trace.json`, `r3-pytest-summary.json`).
- `results` and `picks_dir` are not dates and are outside this commit's scope. Neither is operated on before rejected-date validation completes; no finding is attributed to either.

# What I ran

- Read the narrow commit diff, amended row L/B2, original attacks, both changed test files, the cited corruption tests, and every relevant helper/read/write expression. No network, box, contest account, full fast suite, or tracked-file/index edits.
- `env TZ=America/New_York PYTHONDONTWRITEBYTECODE=1 .venv/bin/python .codex-review/cs2b-confirm/r3_probe.py`: nine current hostile-object cases, four old-source counterexamples, 1,352 exact strings, real signal child, and source-equivalence assertions. Python 3.12.16; all assertions pass. Scratch src comes from `git archive c9c1e342 src`, never a tracked-tree checkout.
- One frozen pytest-family script, `r3_pytest_family.py`, through `~/bts-runs/yield_run.sh`, with the same TZ/PYTHONDONTWRITEBYTECODE settings and `.venv/bin/python`: current three seasonal test files **399 passed**; old-source regression proof **1 expected failure, 1 passed, 247 deselected**. Race-order and unchanged-assertion checks pass. Pytest cache is disabled; basetemp/JUnit/evidence stay here.
- The first wrapped attempt ran those checks successfully but the sandbox denied nice/setpriority. The identical script was retried with permission outside the sandbox; `r3-pytest-family-priority.log` and its watch log confirm the same results and wrapper exit 0. No script was edited after launch.
- Final tracked-tree/index diffs and `git diff --check` are clean. All probes, scratch state, logs, and this report are under `.codex-review/cs2b-confirm/`.

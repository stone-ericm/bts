# Verdict

**BLOCK** `9cb84536` against `9e0ec28a`.

## Blocking findings

- **A5 accepts malformed JSON as a clean off-day.** A successful response with `{"dates":[],"totalGames":NaN}` makes `bts schedule` print `No games scheduled` and idle. `bts day-outcome` exits 0 with `no_games` and sends no DM. The collector also writes an empty JSONL. `Infinity` and `-Infinity` get through the same decoder. At `src/bts/scheduler.py:291`, pass a `parse_constant` callback to `json.loads` that raises `ValueError` naming the token. The existing exception wrapper supplies the schedule context. With just that change simulated at the decoder, the same bytes made the scheduler and collector exit 1 and the watchdog report `schedule_unverifiable` and attempt its DM. `{"dates":[]}` still idled cleanly. [Observed](schedule-head.log) · [Fix control](schedule-strict-control.log)

## Notes

- The independent live status reader, orchestrator's late-reader exception handling and the simulator's historical horizon are existing code. Their behavior is detailed below.
- The new pick tests cover a real fallback refresh and delivery guard. They have less coverage of the scheduler's final fallback call site and the CLI catch-up branches. The separate review probes exercised those branches.

## 1. A1 — Day count and policy

- Enumerated all 730 dates in 2026–27. Every 2026 count matched `max(0, (date(2026, 9, 28) - day).days)`. The per-date slices of both shipped tables and 249,660 executable reach-57 lookups across streaks, saver states and quality bins matched base. Health and strategy agreed on all 730 dates. The head counts are Mar 24 = 187, Sep 26 = 1, Sep 27 and later = 0. The healthy-artifact 2028 check still WARNs and uses Sep 28. [Probe](probe_calendar.py) · [Output](calendar.log)
- Compared 6,800 inputs on Sep 17–26 with streaks 40–56, every trusted best from active streak through 57, both saver flags, two executable legs and `.96` or `.60` for both. The log lists every changed `(date, streak, best, saver, probability)` and old/new action. At `.96`, 454 changed. At `.60`, 956 changed, including Sep 25 at streak 55 and best 55–57 with either saver flag (`double → skip`). Several other days and streaks also move to skip.

| Base → head | .96 | .60 | Assessment at the corrected horizon |
|---|---:|---:|---|
| Skip → double | 336 | 316 | A new day can make the season best beatable or make 57 reachable on the all-double path. The corrected table plays both legs. |
| Skip → single | 4 | 24 | The corrected table plays one leg. On the finale at 56 it can reach 57 with one hit. |
| Double → single | 52 | 332 | The extra day makes the corrected table favor keeping the streak on one leg for these states. |
| Single → double | 52 | 46 | At each probability, four are tail states and the rest switch to reach-57. Both tables can favor advancing the live streak by two. |
| Double → skip | 6 | 30 | The corrected reach-57 table uses today's slot to preserve the streak for a later slate. |
| Single → skip | 4 | 208 | Same reach-57 choice with the extra slate. |

- The changed actions agree with the shipped objectives and corrected day indices for each state. The fitted quality bins model a population of slates. `.96` selects the highest reach-57 bin, whose fitted rates govern the decision. The tail artifact uses one bin. [Policy transition and lookup](../../src/bts/simulate/tail_policy.py#L97)
- `strategy._mdp_action_from` and `_regime` share `_days_remaining`. Health delegates to it and stays CRITICAL for bad artifacts Aug 30–Sep 26 inclusive. `simulate/mdp.py` and `simulate/pooled_policy.py` consume a numerical horizon and switch bins at `late_phase_days`. `split_by_phase_pooled` uses the last N **observed** dates by season, and OPE/Monte Carlo consume supplied horizon or profile steps. The sparse-profile probe placed Sep 21 and 23 in the last two observed dates while the live horizon was 6 and 4. The live path has one date subtraction and one shared health window. [Phase probe output](calendar-phase.log)

## 2. A2 — Same-cycle delivery

- Ran real scheduler DM and public lineup checks, a real in-loop fresh fallback, `run`, `orchestrate` and both CLI catch-up branches. The matrix captures the first outcome before a later scheduler check can overwrite evidence. Catch-up used a real prior saved pick and in-progress game status. Same-cycle mismatch tests used actual state files and the comparison interrupt used the real call site. [Probe](probe_pick.py) · [Base](pick-base.log) · [Head](pick-head.log)

| Final read or comparison | Scheduler DM/public/fallback | `run` / `orchestrate`, including catch-up |
|---|---|---|
| Equal DATED, automatic → automatic with streak 0 → 7 | Sends once, marks attempted and confirmed, locks, writes delivered decision | Posts once and sets Bluesky flags. No scheduler marker, lock or decision record. |
| DATED streak, best or saver change, either automatic/dated transition, missing/invalid/non-required stamp | Refuses, saves no attempt, keeps the original choice and check telemetry, leaves lock and decision clear, dispatches one CRITICAL `contest_state` DM subject to dedup and transport | Exits 1 with `no pick:` on stderr, no post or flag, retains the saved pick, no contest-state DM by the existing CLI contract |
| Midnight changes dated identity or drops it to automatic | Refuses before the second cutoff can authorize a send | Same refusal in the live post branch. The probe held the ordinary window gate open to isolate the final read. Catch-up goes straight to the comparison. |
| Final observation corrupt, absent or a directory | Existing reader rejects, one CRITICAL alert attempt, no send/attempt/lock/commit | `run` prints `no pick:` and exits 1. `orchestrate` propagates `ContestStateError` from the final reader, exits nonzero with the error, and sends no DM. Both preserve the saved pick. |
| `KeyboardInterrupt` or `SystemExit` during comparison | Propagates before mark/send/lock/decision. No alert attempt is reached. | Click aborts or exits. No post, flags or DM. |

- Base sent on changed valid observations and invalid synthetic fresh stamps on the fresh paths. Saved catch-up already refused. Final-reader failures already failed closed. In scheduler midnight cases the base could independently fail the submission cutoff, while the head recorded the earlier contest-state refusal. On the CLI paths, the post-window control exposed the base's obsolete post at midnight.
- A production fresh required choice gets `chosen_on` from the same accepted dated/automatic snapshot used by strategy (`src/bts/cli.py:1240`, `src/bts/orchestrator.py:307`). Those readers enforce the valid types, range and identity. I found no production fresh path that omits it. Preview stamps `required=False` and never posts. Shadow selection is unstamped and writes only a shadow artifact. A legacy cached pick or preview reached by catch-up already refused on base. Dry-run and closed post branches retain their behavior.
- `run` already requires contest state, including with model-only `streak.json`. `orchestrate` with both requirement flags off is already rejected by the CLI public-delivery gate (exit 2). Direct optional calls can still post using model state. Private/research scheduler mode still records `private_locked` without outward picks. Cold live startup already blocks: the scheduler attempted morning and live alarms, `orchestrate` DMed selection failure, and standalone `run` wrote stderr. [Mode and cold-start cases](pick-head.log)
- In the repeated-correction run, two lineup checks, in-loop fallback and final fallback all refused with the pick unattempted and unlocked. Dedup produced one contest-state DM, then the existing missed-pick DM. Next-morning watchdog exited 1 with `no_decision`. With stable state, a later successful fresh check replaces the pick. A saved-only retry keeps refusing until its identity matches or a fresh selection can replace it. A catch-up pick whose game has started can remain stuck for that slate under the existing lock rules.
- The automatic-to-automatic control sent the double selected at streak 0 with the final text `Streak: 7` and retained the streak-0 stamp. [Captured transport](pick-transport.log)
- On the configured box, the 08:30 `day-outcome` watchdog DMs `no_decision` when no other path recorded the day's result. The entry checker skips uncommitted picks. Standalone CLI use has stderr only for a same-cycle refusal unless an independent watchdog is installed. Scheduler stale-automatic, missed-pick, cutoff and all-void alerts have separate triggers.

## 3. A5 — Schedule responses and callers

- Error object or missing/null/string/object `dates` → `ValueError: MLB schedule dates must be a list`. Top-level array → `MLB schedule must be a JSON object`. Non-object date → `MLB schedule dates[0] must be an object`. Missing/null/non-list `games` → `MLB schedule dates[0].games must be a list`. A non-object game after a filtered All-Star game → `MLB schedule dates[0].games[1] must be an object`.
- Bad JSON and zero bytes raise `ValueError: Invalid MLB schedule JSON: …` with decoder location. A 1,500-deep top-level list was decodable and failed the object check. A 10,000-deep array under an object raised a normalized `ValueError` with the recursion error as cause. Valid `dates:[]`, `games:[]` and a slate containing only non-regular games returned `[]`. Mixed `A/E/R` preserved the regular game and still tolerated a missing `gameType`. [Probe](probe_schedule.py) · [Base](schedule-base.log) · [Head](schedule-head.log) · [Depth probe](schedule-recursion.log)

| Caller | Result on the new exception |
|---|---|
| `scheduler.run_day` morning fetch, `bts schedule` | Propagates, exit 1, no off-day idle, no lineup or pick work. Parser errors get one request. Transport failures retain retry behavior (three attempts and 5/10 s waits for `URLError`). |
| `scheduler._next_day_wakeup` | Logs the failure and hands off in 15 minutes. |
| `lineup_collect.collect_for_date`, `bts data collect-lineup-times` | Propagates before polling or writing JSONL. Installed five-minute cron discards its output. |
| `day_outcome.gather`, `bts day-outcome` | Valid intent and readable evidence yield `schedule_unverifiable`, exit 1 and a DM attempt. Failed DM leaves that result and logs the failure. Invalid intent still takes precedence. |
| Independent delivery-status reader | Error object/missing status or malformed bytes still allow an otherwise valid pick. Two void legs refuse and alert. One void leg still sends the intact double. |
| Independent `model.predict._fetch_game_slots` | Error object, missing `games` and empty `dates` yield no slots. `dates:null` still raises `TypeError`. |

- Base treated missing `dates`, some non-list `dates` and missing `games` as clean empties. Those replies now fail in the applicable callers. `tests/test_scheduler_eve_of_break.py::TestFetchScheduleGameType` represents the All-Star Game inside `games`; the July 12 incident audit documents zero-game July 13 and the July 14 `A` game. The fixtures and audit support both off-day shapes tested. I found no raw MLB capture of a date entry with omitted `games`; compatibility for such a live reply is unverified.
- Four simulated systemd launches refetched the same malformed body and exited without DMs. The unit restarts after 30 s plus startup and request time, so journald gets a traceback each launch. The five-minute external heartbeat monitor's real churn predicate flagged +3 restarts in 20 minutes and would ping `/fail` if configured. The independent 08:30 watchdog attempts its own DM when the prior slate is still unverifiable. No actual service restart was run.

## 4. Edited existing tests

| File | Verified edit |
|---|---|
| `test_saved_pick_paths_cs5a.py` | DATED `run` and same-cycle best-only correction in DM/public now assert refusal and preserve the first-read stamp. Matching-identity scheduler control preserves the sent/marked/locked/recorded assertions. Automatic `run` still asserts the posted streak. |
| `test_season_end_date.py` | Updated boundary, opener and finale/next-day expectations as A1 requires. October zero assertion remains. |
| `test_alert_status_cs6b2_carbon.py` | Required stamp added via the actual fixture reader; assertions unchanged. |
| `test_stale_delivery_cs4c.py`, `test_stale_delivery_cs4c_carbon.py`, `test_stale_delivery_cs4c_review.py` | Valid stamps in shared fresh fixtures and corresponding call sites only. Delivery, cutoff, dedup, alert-failure and warning assertions remain. |
| `test_saver_cs3.py` | Aug 29 → 30 for the policy context only. Reserved saver still changes the real result from skip to double. |
| `test_saver_cs3_review.py` | Aug 23 → 24 for the policy context only. The Aug 23 use/undo, health and dashboard checks remain. At 16, saver still changes single to double. |

The eight files match the handover after checking the test diff and assertions. No unrelated assertion was weakened or made unreachable.

## 5. New tests and mutations

- New files on the archived base yielded **36 failed, 23 passed**. Calendar (7) failed all three bundled action replays and all four artifact-severity cases at the old boundary. Pick (16) failed the five late-correction cases, the in-request race, and the best/saver, kind transition, stamp, midnight and interrupt edges because the fresh guard was bypassed. Schedule (13) failed both malformed-body integration cases and all 11 location checks through silent empties, different exception types or unnormalized parser messages. [Full failing node list](pytest-base.log)
- One process ran the three new files, `test_lightgbm_optional.py`, `test_web_saver.py`, and all eight edited existing files. The first pass hit one sandbox-denied localhost bind. The approved rerun passed with **540 passed, 4 skipped**. [Combined run](pytest-combined.log)
- Each wall-clock run passed all **59** new tests. Separate process-local wall clocks started at 18:25 ET on March 24, 2027 and March 24, 2028. The tests' own scenario clocks stayed pinned. [2027](pytest-wall-2027.log) · [2028](pytest-wall-2028.log)

| Mutation run against the relevant new module | Result |
|---|---|
| Restore fresh exemption on DM lineup, public lineup, in-loop fallback, `run` ordinary post, `orchestrate` ordinary post | Killed separately, with 2, 1, 11, 1 and 1 failures respectively |
| Restore fresh exemption on each CLI catch-up branch | Both survive. Real locked selection results already have `sel.decision is None`, so restoring the exemption leaves saved catch-up checking in place. |
| Restore fresh exemption on final fallback only | Survives (31 passed). Applying the same scratch mutation to the full-day review probe sent and committed the stale final-fallback pick after the earlier refusals. Head refused all four attempts. [Proof](pick-final-fallback-mutant.log) |
| Compare only streak for dated pairs | Killed by the best-only and saver-only cases |
| Refuse automatic → automatic | Killed by the changed-automatic delivery control |
| Revert the 2027 boundary / make the 2026 count inclusive | Killed by seven calendar cases / the full-year 2026 invariant |
| Return `[]` for non-list dates / skip validation of non-object games before filtering | Killed by four schedule cases / the malformed game following a filtered game |

Eleven of 14 isolated mutants were killed. Add the final-fallback probe and a malformed JSON constant case to the regressions. The parser's constant case is a production failure as described above. [Run exit codes](pytest-job-status.log)

## 6. A3, A4 and scope

- `git diff` shows the scheduler changes confined to the hydrated parser and the delivery comparison. The `run_single_check` return, overrun reuse, and loop wake logic are byte-identical. Both archived Slice 3 controls ran against base and head. A3 still started a second cascade at 19:00 and recorded cutoff refusal. A4 sent with one game and stranded the two-game pick at the 19:25 check. [Base](scope-base.log) · [Head](scope-head.log)

## What I ran

- Archived `9e0ec28a` with `git archive 9e0ec28a src scripts`. Ran `probe_calendar.py`, `probe_pick.py` and `probe_schedule.py` with `.venv/bin/python`, pinned ET inputs and captured transports. Ran the original A3/A4 regressions from Slice 3 with the fixture root pointed at this checkout. No tracked files changed.
- Ran the focused head/base, two wall-clock and 14 source-mutant pytest jobs serially through `~/bts-runs/yield_run.sh` after 01:14 ET using `probe_test_matrix.py`. The approved web-saver test used a loopback socket. No external network requests or live service operations were used.

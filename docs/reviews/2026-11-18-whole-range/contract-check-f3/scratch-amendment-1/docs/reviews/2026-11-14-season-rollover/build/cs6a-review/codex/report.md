# Verdict

**BLOCK** on row 4. Commit `cad7c5ec` closes the unchecked-state launch paths exercised here. The dry-run command can change the date it schedules and enter a different worker path.

## Blocking findings

- **B1 — Omitted dry-run date moves across ET midnight.** Run `bts schedule --config C --dry-run` with `[scheduler]` containing `contest_state_required = true` and omit `--date`. Start at 23:59:59.900 ET on July 15, 2027, and finish the TOML load at 00:00:00.100 ET. The base fetches July 15, sees a game, writes a `running` heartbeat, and returns. At `src/bts/cli.py:2551` and `:2559`, the branch loads and checks first, then computes the date. It fetches July 16, finds an off day, looks ahead to July 17, leaves a `sleeping` heartbeat, and attempts to sleep for 35,999.9 seconds. Both runs exit 0 with captured sleep. No warning applies to this configuration. `midnight_probe.py` drove the real loader and `run_day` with the same clock and schedule fixtures on both revisions.
- **Smallest fix.** Resolve the default ET date before loading TOML when `dry_run` is true and `date` is omitted. Keep the non-dry default-date computation after the gate. The scratch repair produces byte-identical midnight evidence to the base and still refuses a non-dry DM launch before the default date, worker, or resource access.

## Notes

- `bts entry-intent --help` still describes agreement with the delivery mode as sufficient at `src/bts/cli.py:1764`. An agreeing DM/enter file without either required-state setting now exits 1. Mention the startup check in the help.
- The frozen evening clock in `test_schedule_defaults_to_eastern_date_not_utc` covers the ET/UTC boundary and warning, and leaves the new ordering across ET midnight untested. `midnight_probe.py` exercises that boundary.
- The tracked kickoff and earlier reviews describe the example's then-current false values as historical context. The Pi5 installer tells operators to copy the file without asserting its values. I found no conflicting active consumer of the example.
- Contract assessment. I found no row that requires a contract correction. The dry-run worker may idle on an off day as specified. B1 changes which day reaches that worker.

## Can a new launch pick outward on unchecked state

For well-shaped TOML, a scheduler launch that can send by DM or to the public feed, or declares exact `enter`, must require contest state via the truthiness of either flag. The orchestrate command always counts as public. A non-dry `bts run` must use a required read at every decision-state read site. A required read needs today's effective dated observation or automatic state covered by the active season switch.

`matrix.py` exercised 65,559 distinct real TOML files per revision through all four commands, with worker spies. The independent expectations also matched 655,590 branch predicate calls, including delivery overrides and malformed sections. The CLI changes were 20,657 schedule refusals, the same number of dry-run warnings, 31,283 orchestrate refusals, and 20,631 entry-intent changes. The matrix used fixed explicit dates. The omitted dry-run date exposes B1.

`startup_probe.py` passed last season's uncovered `auto_v1` through real scheduler selection and delivery for all 8,164 accepted public and 7,344 accepted DM mappings, and through real orchestration for 34,276 accepted mappings. None selected or sent a pick. The scheduler's required reads emitted contest-state alerts. Orchestration's initial refusal could DM the operator that no pick was made. Covered-state controls delivered by DM and by feed.

`pick_delivery = "local"` with `entry_intent = "research"` and both flags absent still uses optional state in the real scheduler. It selected on streak 7 and locked locally with zero pick DMs and feed posts. With the same TOML, `bts orchestrate` at the base publicly posted that pick. The branch exits 2 before cascade. With a truthy flag, the branch accepts the launch and uses the required read. It exits 1 with an error DM on the uncovered fixture, and posts to the public feed on the covered fixture. An outward scheduler mode plus a truthy flag can send a checked pick even if the declared intent says `research`.

## bts run and bts orchestrate

- `pick_probe.py` ran 66 path/state pairs per revision plus three state changes between reads. It covered skip, locked catch-up, an already-posted lock, fresh single and double, delayed posting, no valid candidate, empty predictions, both pipeline exception handlers, and dry run. The real loader read pinned files for no observation, last-season `auto_v1`, a legacy manual override, an in-effect dated observation, a covered automatic observation and a corrupt switch.
- `compare_evidence.py` found 38 cases with matching streams, exit, captured transport and saved bytes after accounting for the intended required-read argument. Empty predictions, pipeline errors and dry runs made zero state reads. All 31 refusals exited 1, with the exact diagnostic prefix `bts run: no pick: ` and the loader's reason on stderr. At the first read only the base prediction and ranking prefix stood. After simulated switch removal before the locked catch-up read, the lock line stood and the existing pick and evolution files were byte-identical. On a post-save refusal the saved, unposted pick and evolution row survived. There were zero saves and posts after any refusal. The same three switch-removal variants completed posting on the base.
- The real orchestrator command loaded exactly once before the configuration swap. The audit trace was read, then the probe's replacement write. The core received the identical mapping and wrote to the checked picks directory. The base trace was write, then read, and it wrote to the replacement directory. Missing and unreadable paths exit 2 at Click on both. Malformed TOML produces `TOMLDecodeError` and exit 1 on both. A false worker result still maps to exit 1.
- The production CLI is the only tracked caller of the core found in `src` and `scripts`. The two-positional test call and the historical caller harness retain path loading and bypass the launch gate. Ran a two-positional call on both revisions with private/research TOML, seeing one load and the same captured public post. Scheduler `run_and_pick` callers use a distinct function.

## Entry intent, cron installer and day-outcome

- Problem order matched the base order with one append by default. A malformed scheduler still yields its single original problem. Opt-out matched the base in every matrix cell. The only production consumers of the resolver found were the entry-intent command and `day_outcome.gather`.
- With valid DM/enter and neither flag, real Hetzner `show` and `install` exit 1 with empty stdout, the new problem as the last entry-intent line, and the installer's generic error immediately after it. The stand-in `crontab` was never invoked and the prior crontab was byte-identical. Each flag on its own restored base-equivalent stdout, stderr, exits and installed lines. All three `remove` variants matched the base.
- Real `day_outcome.gather` and CLI matched the base classification, stdout, stderr, alert text and exit across 240 cases. These covered confirmed and unresolved entry, deliberately skipped and private commits, missing decisions, no games, voids and schedule errors across valid, disagreeing, absent and malformed intent configurations.

## Tests and false greens

- All five added or touched files passed at each fake ET clock, **192 passed** on 2027-07-15 and **192 passed** on 2028-07-15. Unchanged day-outcome files passed **91** controls. The base comparison of 107 selected nodes yielded **43 failed, 64 passed**. Twenty failures were availability of the new predicate or opt-out signature.
- `base-green-controls.txt` names the 12 added controls that pass against the base. They cover existing invalid-intent/delivery ordering and malformed-scheduler handling, schedule mapping identity, malformed TOML, direct orchestrator compatibility, one load through the real core, dated and covered state, empty/dry early returns, cron removal and required-state cron parity. The real-core load-count control also passes when the base loads inside the core. The command spy, reload mutant and swap probe distinguish the location and identity of the checked snapshot.
- All 16 isolated mutants failed the expected tests. This includes dropping the schedule, orchestrate or entry-intent gate, bypassing the resolver append, ignoring intent/delivery/override, accepting either malformed section, making each of the three `run` reads optional, reloading orchestration, enabling the gate in day-outcome and turning either example flag off. The day-outcome mutant is caught by the unchanged day-outcome tests. Worker, loader, save and post spies patch the symbols looked up by production.
- Migrated `TestBtsRun.test_run_saves_pick_and_posts` retains the pick/save/post assertions on a pinned, covered zero-streak fixture. `test_run_uses_fresh_contest_streak_for_live_action` retains real model streak 4 versus contest streak 7, saver unavailable, and the required firing of the policy assertion. `test_schedule_defaults_to_eastern_date_not_utc` retains the ET date assertion and adds the exact warning.
- Migrated `test_agreeing_intent_and_delivery`, `test_enter_install_adds_only_the_entry_cron` and `test_the_day_outcome_check_is_installed_under_either_intent` retain each original resolver and cron assertion with a required setting in the enter fixtures. The historical caller test `test_real_selection_fallbacks_orchestrator_and_clean_outcome` preserves the forced refusal and other caller assertions. Its former successful run and post assertions are replaced by exact refusal and zero posts as the contract directs. The historical probe source is intact.

## Rows 1 to 9

1. **Holds.** `matrix.py` and `compare_evidence.py` confirm shape order, truthiness, aliases/conflicts, exact intent, override and no resource or clock access after imports.
2. **Holds.** `matrix.py` and `diagnostics_probe.py` confirm ordered diagnostics and the opt-out, all installer actions and day-outcome parity.
3. **Holds.** `startup_probe.py` records one TOML open, no default date or worker, no other fixture-resource opens, and the same tree on refusal. Acceptance passes the original mapping.
4. **Fails.** `startup_probe.py` shows same-date dry-run parity on a game day and an off day after the warning. `midnight_probe.py` finds B1 with an omitted date.
5. **Holds.** `startup_probe.py` swaps the file after the gate, traces opens, checks direct calls and load errors, and runs public and private configurations through the real core.
6. **Holds.** `pick_probe.py` and `compare_evidence.py` verify all three required reads, their exit/streams and their respective artifact boundaries.
7. **Holds.** `pick_probe.py` shows all dry-run cases read zero state and match base streams and files.
8. **Holds.** `diagnostics_probe.py` and Git diff confirm both literal true flags, exact `enter`, gate acceptance and the requested comments as the only other edits.
9. **Holds.** The scoped source diff and direct-call/caller probes preserve the named core helpers, scripts and existing entry-point behavior outside rows 1–8.

## What I ran

- Compared `09d9b51f` archived under `base/` with `cad7c5ec` using `matrix.py`, `pick_probe.py`, `startup_probe.py`, `diagnostics_probe.py`, `midnight_probe.py` and `compare_evidence.py`. The corresponding logs and JSON traces are beside the scripts. I also ran the midnight and startup probes against the minimal repair in `testbed/src` after the test suite completed.
- Ran `pytest_harness.py` through one `yield_run.sh` invocation and waited for wrapper exit 0. The warning about setting niceness did not stop the run. Logs cover both clocks, the base, unchanged day-outcome controls and all mutants. Probes and Python children used the network audit guard and captured transports. The cron shim used a throwaway HOME and stand-in executables.
- `git diff --check 09d9b51f HEAD` passed and the tracked tree is clean. The testbed and the repair are review artifacts.

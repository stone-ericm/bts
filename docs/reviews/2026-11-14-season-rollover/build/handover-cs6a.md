# CS6a handover

Branch `sb-cs6a` is based on `09d9b51f4a7576ceff9e4e36cb3db29471335c88`. Commit `cad7c5ec477c11a20adaccb0a7c725b5c1d0d513` implements CS6a and its tests. The branch was not pushed or deployed. The final gate ran on that commit with an empty tracked diff.

## Contract and evidence

The startup check applies to configuration mappings as `tomllib` produces them and to the command-line entry points below. The predicate tests also exercise a Python-only `None` and explicit delivery overrides. The required decision reads enforce the existing CS4b-2 season coverage at each call, using the state snapshot that call reads.

| Row | Implementation | Evidence |
| --- | --- | --- |
| 1 | `src/bts/entry_intent.py:31` validates scheduler then health table shape, honors the existing truthiness rule, and identifies all outward-delivery and exact-enter conditions. | `tests/test_live_gate_cs6a.py:65,98` enumerate aliases, precedence, flag and intent values, malformed sections and overrides, with file and clock spies. The configuration comparison exercises these rules through real TOML. |
| 2 | `src/bts/entry_intent.py:69,93` appends the gate after legacy problems. `src/bts/cli.py:1763` keeps the CLI protocol. `src/bts/day_outcome.py:157` opts out so declared intent and delivery still drive reports. | `tests/test_live_gate_cs6a.py:147,158` checks order and opt-out. `tests/test_entry_intent.py:228,242,250` runs the unchanged installer with a local crontab shim. The gate is the CLI's last problem, then the installer prints its generic refusal, as required by Amendment 1. Both refusal actions leave the existing crontab byte-identical without invoking `crontab`. Remove works, and both day-outcome files passed unchanged. |
| 3 | `src/bts/cli.py:2540,2551` loads once, then refuses before default-date resolution and worker access. | `tests/test_live_gate_cs6a.py:166,206` captures exact stderr and exit, checks no picks, state, heartbeat, schedule, prediction or date access, and checks mapping identity on acceptance. |
| 4 | `src/bts/cli.py:2551` warns first and passes the same configuration to the scheduler with `dry_run=True`. | `tests/test_live_gate_cs6a.py:220` exercises real game-day planning and off-day heartbeat, notifications and idle control flow with fetch and sleep captured. `tests/test_cli_schedule_date.py:15` still verifies the ET date. |
| 5 | `src/bts/cli.py:2514` checks the public delivery override, then passes the checked mapping to `src/bts/orchestrator.py:320,330`. | `tests/test_live_gate_cs6a.py:259,272,281,354,369,389` checks one load and object identity, false to exit 1, invalid mode refusal, malformed TOML error class, a real public post on the base with private/research and stale uncovered state, and the retained direct-Python behavior and accepted CLI posting. |
| 6 | `src/bts/cli.py:1161,1210,1255` each pass `require_contest_state=True` and catch only `ContestStateError` there. | `tests/test_live_gate_cs6a.py:422,477` runs the real loader against pinned files for first-read skip, locked and fresh-double paths and later locked and post-save reads. It asserts exact stderr and exit, required arguments, saves, posts, and the surviving unposted pick and lineup evolution. Dated and covered automatic controls post. `tests/test_season_contract_cs4b2_edges.py:218,253` repeats the historical injected refusal with zero posts. |
| 7 | `src/bts/cli.py:1132,1156` retains the empty prediction and dry-run exits before decision-state access. | `tests/test_live_gate_cs6a.py:499` spies on the state reader, saver and poster. The existing CLI dry and empty controls still pass. |
| 8 | `config/orchestrator.example.toml:7,41` declares `enter`, sets both flags to literal `true`, and documents the required state and launch refusal. | `tests/test_live_gate_cs6a.py:291,300,304` parses the example and asserts each flag independently, the intent, and both verification results. |
| 9 | The scheduler helpers, `run_day`, `run_and_pick`, loader, health checks, preview, dashboard and cron scripts match the base. The core orchestrator's only behavioral change is using a supplied snapshot, and the only day-outcome edit is the opt-out. | The committed diff and the original scheduler, orchestrator and day-outcome suites, direct two-argument orchestration with actual saved/post state at `tests/test_live_gate_cs6a.py:369`, and the preserved callers tested in `tests/test_season_contract_cs4b2_edges.py:218`. |

## Archived-base comparison

`git archive 09d9b51f src` supplied the base. `.codex-review/cs6a/compare.sh` ran the 5,760-cell cross-product, including cases whose inaccessible keys produce identical TOML when a section is absent or malformed. It recorded raw results in `base.jsonl` and `branch.jsonl`. Each command was invoked through Click using real TOML, with scheduler and orchestration workers replaced by spies. The verifier checked exact status, stderr, stdout and worker call counts for every cell. Every difference matched its row.

| Observation | Changed or new problem cells | Reason |
| --- | ---: | --- |
| `live_gate_problem` | 4,448 | Row 1 adds the predicate. It reports 3,200 malformed-section cases and 1,248 otherwise well-shaped live cases with neither flag truthy. The other 1,312 cells return `None`. |
| `live_gate_problem` with the public override | 4,640 | Rows 1 and 5 use the same shape rules and public delivery regardless of effective scheduler delivery. The other 1,120 cells return `None`. |
| `check_entry_intent` | 2,528 | Row 2 appends the gate after existing problems. The 1,920 malformed-scheduler cells retain their single existing problem. |
| `bts entry-intent` | 2,528 | Row 2 adds the last stderr problem. Of these, 144 former successes now exit 1 with no stdout and 2,384 still exit 1. Unchanged cells include 144 successes and 3,088 failures. |
| `bts schedule` | 4,448 | Row 3 adds exit 2 and the refusal with zero worker calls. Accepted cells retain exit status, output and worker calls. |
| `bts schedule --dry-run` | 4,448 | Row 4 adds only the warning. Worker calls and exit status match the base for every cell. Real worker side effects were tested separately. |
| `bts orchestrate` | 4,640 | Row 5 adds exit 2 and refusal before the worker. Accepted cells retain the status and call count, and the targeted tests verify that the worker receives the exact checked mapping. |

An additional comparison ran the real Hetzner `show` and `install` commands for a valid DM/enter configuration with required state. Stdout, stderr, exit status and the crontab contents matched the base on both actions, including all 21 installed lines.

## Existing test changes

- `tests/test_cli_integration.py:136` (`test_run_saves_pick_and_posts`) now pins the operation clock in ET and supplies a covered automatic zero-streak observation to retain the successful saved/post path.
- `tests/test_cli_integration.py:217` (`test_run_uses_fresh_contest_streak_for_live_action`) pins the operation clock in ET and replaces the legacy manual observation with a covered automatic one. The test still distinguishes contest streak 7 from model streak 4 and checks that the saver is unavailable.
- `tests/test_cli_schedule_date.py:15` (`test_schedule_defaults_to_eastern_date_not_utc`) adds an exact dry-run warning assertion and retains the date assertion.
- `tests/test_season_contract_cs4b2_edges.py:218` (`test_real_selection_fallbacks_orchestrator_and_clean_outcome`) retains the forced refusal, asserts exit 1, the injected message and zero posts, and keeps the other caller assertions and historical probe source.
- `tests/test_entry_intent.py:35` (`test_agreeing_intent_and_delivery`) adds a required-state flag in the three positive enter cases, for DM, the Bluesky alias and implicit public delivery.
- `tests/test_entry_intent.py:160` (`test_enter_install_adds_only_the_entry_cron`) adds a required-state flag in its positive enter install.
- `tests/test_entry_intent.py:340` (`test_the_day_outcome_check_is_installed_under_either_intent`) adds a required-state flag in its enter case so it still verifies the identical day-outcome cron under both intents.

## Launch and delivery inventory

| Entry point | CS6a behavior |
| --- | --- |
| `bts schedule` and the scheduler systemd unit | Applies the startup gate to new launch configurations. The scheduler can send DMs and post publicly, or lock privately. Dry-run only adds the warning. |
| `bts orchestrate` | Applies the startup gate as public delivery. Skip, catch-up and new-pick posting all use the public feed. |
| `bts run`, by hand or from legacy Mac cron | All three decision reads on the non-dry path are required. Skip, catch-up and new-pick posts are withheld after a refusal. A refusal on the post-save read can leave the already-saved unposted file and lineup evolution record. Dry runs and empty prediction results still exit before state reads. |
| `bts entry-intent` and Hetzner cron `show` / `install` | Reports configuration issues at install time. It sends or enters no pick. The installer exits before touching existing jobs; `remove` bypasses the config check. Pi5 cron installs grading only. |
| `bts preview` and the dashboard | Optional decision state. Preview can write a pick file, and the dashboard can display it. Cached preview delivery without re-selection is deferred to CS5. |
| `bts check-pick-entered` | Can DM committed pick names. It checks the entry and sends reminders, with human submission outside this gate. A config edit leaves an already installed entry-check cron in place. |
| `bts day-outcome`, reports, grading and result replies | Existing diagnostics and result processing continue, including diagnostic DMs and public replies about existing decisions. These paths issue no fresh recommendation transport. |

No supported command was identified that submits a pick to the contest automatically. `run_day`, `run_single_check`, delivery helpers and the two-argument orchestrator function can be called directly from Python and bypass the launch gate. They retain their prior required-state settings and decision behavior.

## Operational limits

- A running process keeps its loaded configuration. The launch gate leaves committed decisions and in-flight sends in place. Private research can still pick and lock locally with optional state.
- The refusal goes to stderr. A DM scheduler launched without either setting exits 2 before READY or heartbeat. Systemd may restart it every 30 seconds or stop on rate limiting, and its end-of-day health check cannot run. A blocking restart in the deploy workflow can fail before canary or rollback and leave the new commit installed. A canary rollback can restore a revision without the gate. Independent five-minute heartbeat cron and next-morning day-outcome can surface symptoms if installed and routed. This change guarantees neither notification nor rollback.
- Before deploy or the first start of a restored box, set both literal flags in the service’s actual TOML, confirm it with that revision’s `bts entry-intent`, verify or install the intended cron, and inspect `bts season-contract status`. The remaining change sets must land before deploying this season-boundary work.
- `bts schedule --dry-run` still writes heartbeat, calls notification helpers, fetches the schedule and can idle on an off day.

## Verification runs

All pytest runs used `yield_run.sh`, sequentially, with America/New_York and no pytest cache. The initial fast suite exited before any test files were created. The final fast suite was run alone on the committed tree.

| Run | Result |
| --- | --- |
| `baseline.log` | 8,103 passed, 37 failed, 16 errors, 19 skipped, 6 deselected, 1 xfailed. The 53 failure/error nodes exactly matched `~/bts-runs/gate-failed.txt`. |
| `red.log` | 968 passed, 41 failed, 8 skipped, 1 xfailed. The gate function and opt-out keyword accounted for 20 availability failures. The other red assertions exercised missing behaviors, except the game-day fixture initially lacked a `teams` object. Existing accepted and direct-call cases passed. |
| `red-fixture.log` | 5 expected failures, 44 deselected, while still on the base. The corrected game-day case and the off-day case reached the missing-warning assertion after verifying worker behavior. Both flags and the example intent failed independently. Across the two base runs, the final test set has 23 behavioral RED cases after excluding the 20 availability cases. |
| `green.log` | 1,011 passed, 8 skipped, 1 xfailed. The same targeted selection covers the CLI, entry intent, cron shim, day-outcome, direct scheduler and orchestrator tests, and other call sites identified by source search. |
| `clock-2027.log` | 192 passed with the Python wall clock set to 2027-07-15 ET. All five added or edited test files ran. |
| `clock-2028.log` | 192 passed with the Python wall clock set to 2028-07-15 ET. Same five files. |
| `compare.log` | 5,760 cases matched, plus accepted cron parity. |
| `final.log` | 8,156 passed, 37 failed, 16 errors, 19 skipped, 6 deselected, 1 xfailed in 26m 18s. Exactly the same 53 failure/error nodes as the machine list and the baseline. No new failures. |

Logs and replay helpers are in `.codex-review/cs6a/`. The linked Git index required an approved `yield_run.sh` invocation to commit the ten scoped files. The work used local fixtures and captured transports. No deployment, live box, contest account or real crontab was used.

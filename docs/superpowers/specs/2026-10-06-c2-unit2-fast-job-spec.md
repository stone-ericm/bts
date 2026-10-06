# C2 (a) unit 2 spec: the five-minute job (W1, W5, W7)

**What this is:** the spec for C2 item (a), review unit 2 (C2 proposal §5 (a)). It covers these registration rows of `docs/sota_audit/2026-10-04-prereg-c1-watchdog.md` (FROZEN):
- **W1:** W-deliver, W-postponed, W-decision and W-mode;
- **W5:** W-restart;
- **W7:** W-self;
- the §3 schedule rules for the five-minute job.

**Status:** DRAFT for spec review (at most 2 Codex rounds), then code (at most 2 rounds). Production code: reviewed until SIGN, and deployed only with the whole watchdog under Eric's D7.

**Built on:**
- **W0,** signed in C2 unit 1 (`f1b948a`, branch `c2-watchdog`): the owned root, the registered `<job>/<check_id>` identity, notifications, and the confinement gate with no process creation.
- **The C1 W1 draft** (`docs/superpowers/specs/2026-10-06-c1-r2-w1-spec.md`, unreviewed) and its evidence map (`…-w1-evidence-map.md`). This spec supersedes the draft; changes from it are marked **C2:**.

**Not in this unit:** W2 (entry) and W4 (grade) are unit 3. W6 (restore), W8 (capture-stop, with rank-1 Part A) and W9 (coverage matrix) are unit 4. W3 (W-state) stays deferred with P3. The cron line and the deployment come with the whole watchdog's D7.

## 0. Common rules

### 0.1 The job
- **Registration:** `register("fast", [("deliver", …), ("postponed", …), ("decision", …), ("mode", …), ("restart", …), ("self", …)])`. Checker identities are `fast/<id>`; the business check names are `W-deliver`, `W-postponed`, `W-decision`, `W-mode`, `W-restart` and `W-self`.
- **Schedule (R3, §3):** every five minutes, every day and every hour, games or not. It never depends on the scheduler reaching end of day.
- **C2: a whole-job budget of 120 s** (registration §3, "a two-minute execution bound"; W0 left the whole-job deadline to this unit). It is measured with a monotonic clock from job start:
  - each seam call (§0.4, §0.5) gets a timeout of `min(10 s, remaining − 20 s)`;
  - a call that would get less than 2 s is not made, and every predicate that needed it is `unverifiable` ("job budget exhausted");
  - the notifier's flush budget is `max(0, remaining − 5 s)` when it starts;
  - the run's start, end and elapsed time are recorded for W-self (§6).

### 0.2 Target dates
- **Today:** the clock's ET date.
- **Yesterday, always:** the end-of-day skip decision can land after midnight, and the "unresolved day disposition" rule closes at 06:15 ET the next day.
- **Carried dates:** any earlier date within 3 days with an **open** target of one of this job's business checks. They are read from the notification state under its short lock (`notify.load_state`), and that read never writes.
  - If the notification state cannot be read, the targets are today and yesterday only.
  - The run's notification step then reports the unreadable state as an infrastructure error, so W-self pings `/fail`.
- **Labelling:** each result names its own `et_date`.

### 0.3 Inputs: read-only (R1, R5)
| Input | How it is read | When it is unusable |
|---|---|---|
| `data/picks/<D>.json` | one `read_bytes`, then `parse_daily_pick` | a parse failure is `unverifiable` ("pick file unreadable"); never "absent" |
| `data/picks/<D>/decision.json` | one `read_bytes`, then `parse_decision` | present but `None` from the parser is **invalid**, so `unverifiable`, never "absent" (`load_decision` would hide this) |
| `data/picks/<D>/scheduler_state.json` | one `read_bytes`, then `json.loads` (never `load_state`, which renames corrupt files) | unparseable is `unverifiable` |
| archives `data/picks/<D>/{stale_pick,deferred_fallback,refused_delivery}_*.json` | a directory listing plus one `read_bytes` each | an unreadable archive is `unverifiable` for the predicate that needs it |
| `data/.heartbeat` | one `read_bytes`, then `json.loads` | as in §5 |
| the config, `--config` (default `~/.bts-orchestrator.toml`) | `tomllib` | unreadable is a W-mode `fault` (configuration) |
| the crontab | the process seam (§0.4) | as in §4 |

- **One snapshot:** each file is read at most once per run. Different files are read at different instants, so no multi-file consistency is claimed; that is W-state's job (deferred with P3). A predicate that combines files states which reads it used.
- **Writes:** none, except the watchdog's own root (results, notification state, the schedule cache in §0.5, the restart samples in §5 and the self records in §6).

### 0.4 C2: the process seam `bts.watchdog.proc`
This is unit 1's R3-4 rule carried forward. No watchdog code creates a process except through this seam.
- **One function:** `run(name, timeout)`, for a fixed table of exactly five commands:

| Name | argv |
|---|---|
| `crontab` | `crontab -l` |
| `nrestarts` | `systemctl --user show bts-scheduler -p NRestarts --value` (exactly `read_nrestarts`' argv) |
| `unit` | `systemctl --user show bts-scheduler -p ActiveState,SubState,MainPID,ActiveEnterTimestamp` |
| `ps` | `ps -eo pid=,ppid=,args=` |
| `head` | `git -C <repo root> log -1 --format=%H%x20%ct` |

- **How it runs:**
  - no shell; `stdin=DEVNULL`;
  - a minimal environment: `PATH=/usr/bin:/bin`, `HOME`, `LANG=C`, and the `XDG_RUNTIME_DIR` / `DBUS_SESSION_BUS_ADDRESS` defaults that `read_nrestarts` sets;
  - stdout is capped at 64 KiB, and a capped read is an error;
  - a timeout or a non-zero exit is returned to the caller, never raised past the predicate.
- **Unknown names raise.** `read_nrestarts(unit, run=adapter)` gets an adapter that accepts only the `nrestarts` argv.
- **In the gates:** gates 1 and 2 substitute `proc.run` (registration §4.1, "patch … process observations"). Inside gate 2 any process creation still kills the gate.
- **The real seam's own tests,** outside the gate:
  - an unknown name refuses;
  - every allowed argv is exact;
  - a timeout and an oversize output are reported;
  - the environment is minimal.
- **Stated boundary:** the real children are fixed system binaries run read-only. They are outside the kernel write gate.

### 0.5 C2: the MLB schedule seam `bts.watchdog.mlb`
- **One request** per date and use: `GET https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=<D>`, with `urllib`, no retry loop, and the §0.1 timeout.
- **The regular-season filter** is the scheduler's own `bts.util.is_regular_season_game`. So the watchdog expects nothing when the scheduler would pick nothing (postseason, spring).
- **Refresh rate:**
  - today's schedule is fetched every run;
  - any other date reuses a cached response under `data/watchdog/cache/schedule-<D>.json` for up to 30 minutes.
  - That is about 288 + 48 requests a day, comparable to the existing `*/5` lineup collector.
- **On failure:** every predicate that needs refreshed times or status is `unverifiable` ("schedule unavailable"), and the run still completes.
- **Never:** an authenticated request or a cookie read (R6).

### 0.6 Times, statuses and selections
These are carried from the draft.
- **Clocks:** ET-aware via the W0 seam. A pick's `game_time` is UTC, and a naive value is read as UTC.
- **Refreshed first pitch:** the schedule's `gameDate`, falling back to the pick's `game_time` only when the game is absent (that absence is W-postponed evidence).
- **The cutoff:** the minimum over the selected legs of (refreshed first pitch) − `SUBMISSION_CUTOFF_MIN`. Equality counts as past.
- **Statuses:** W0's. **Selections:** `<batter_id>@<game_pk>`, legs joined with `+`, or `day`.

### 0.7 Mode gating (R4)
- **The intent** is `check_entry_intent(config)` (P4, deployed).
- **Under `research`:**
  - W-deliver reports `verified` ("research: no delivery expected");
  - W-decision still runs;
  - W-mode reports any breach.
- **An intent problem** makes W-deliver's delivery expectations `unverifiable`. W-mode reports the configuration fault.

## 1. W-deliver
Under `enter`, for target date D, the draft's §1 table applies unchanged:
- no games;
- delivered before or after the cutoff;
- `locked_unconfirmed` without a receipt;
- MDP skip;
- a pending skip;
- an overdue candidate;
- a refusal;
- a missed early game;
- a deferral;
- no day disposition.

The pre-cutoff risk rule also applies unchanged.

**C2: incident names are fixed** (they are part of the dedup key): `I-late-delivery`, `I-unreceipted-commit`, `I-overdue-candidate`, `I-delivery-refused`, `I-missed-early-game`, `I-no-disposition` and `I-wake-after-cutoff`.

**Text:** post-cutoff faults never say "enter now".

## 2. W-postponed
The draft's §2 applies, with one change.

**C2 (resolves draft question 3): an undelivered candidate on a postponed or cancelled game is `pending` until its cutoff.**
- Text: "candidate on a postponed game; the scheduler normally re-selects".
- **Why (reasoning only):** the scheduler re-selects at its next check, so a fault before then would mostly alert on routine churn.
- **If it persists:** a candidate still undelivered at its cutoff is W-deliver's `I-overdue-candidate`, so the risk is not dropped.

A **delivered or committed** slot on a postponed or cancelled game is a `fault`, as in the draft (`I-postponed-slot`). Every result says "current status at <observation time>".

## 3. W-decision
The draft's §3 applies unchanged, including:
- the 5-minute decision-after-delivery grace;
- the runner-up `double_down` caveat on `single` records;
- `primary=null` is `unverifiable`;
- the 06:15 ET next-day unresolved-disposition deadline;
- an invalid `decision.json` is `unverifiable`, never absent.

**C2: incident names:** `I-decision-missing`, `I-score-bearing-skip`, `I-score-bearing-preview`, `I-selection-mismatch` and `I-unresolved-disposition`.

## 4. W-mode
The draft's §4 applies, with these changes:
- **C2: the crontab comes through the process seam** (resolves draft question 5).
  - **"Installed"** means an uncommented line that carries the `# BTS-HETZNER` marker and contains `bts check-pick-entered`.
  - **`no crontab for <user>`** on stderr with a non-zero exit is an empty crontab.
  - **Any other failure** is `unverifiable`.
- **C2: the configuration is read with `tomllib` from `--config`.** An unreadable or unparseable config is a `fault` "configuration unreadable" (`I-config-intent`); it is never `verified`.
- **Incident names:** `I-config-intent`, `I-enter-without-checker`, `I-research-with-checker`, `I-delivered-in-research` and `I-private-under-enter`.

## 5. W5: W-restart
**Registration row:**
- independently observe scheduler progress and lifecycle, even with a fresh heartbeat or on a no-games day;
- declare the external heartbeat/churn monitor as a dependency and exercise its fresh-heartbeat churn thresholds;
- surface unavailable counters and monitor configuration;
- register dispositions for multiple scheduler instances, absent polling after a restart, and deploy-version observations.

**Reuse:** `is_stale`, `assess_churn` and `read_nrestarts` from `scripts/check_heartbeat.py`, loaded by file path from the repository root (`importlib.util.spec_from_file_location`). That is because the installed `bts` entry point does not put the repository root on `sys.path`. The watchdog never calls its `main()`, which writes and pings.

| Observation | Result (selection `day`, date today) |
|---|---|
| `is_stale(data/.heartbeat, now)` is stale | `fault` "heartbeat stale: <reason>" (`I-heartbeat-stale`) |
| `nrestarts` through the seam, then `assess_churn(samples, n, now)` with the watchdog's **own** samples `data/watchdog/restart/churn.json` (thresholds +3/20 min, +3/60 min, +4/180 min) | churn: `fault` (`I-restart-churn`); counter unavailable: `unverifiable` "NRestarts unavailable" |
| the crontab has no uncommented `# BTS-HETZNER` line running `scripts/check_heartbeat.py` | `fault` "external heartbeat monitor not installed" (`I-monitor-missing`) |
| the heartbeat is fresh, but the monitor's own sample file `data/health_state/scheduler_churn.json` is more than 20 minutes old, or absent (the monitor rewrites it every run while the heartbeat is fresh) | `fault` "external monitor not observed running" (`I-monitor-silent`) |
| `ps` shows more than one scheduler **instance** | `fault` "multiple scheduler instances" (`I-multiple-schedulers`); `ps` unavailable: `unverifiable` |
| `unit`: `ActiveState` not `active` | `fault` "scheduler unit not active" (`I-scheduler-inactive`) |
| `unit`: `ActiveEnterTimestamp` more than 10 minutes ago, and the heartbeat timestamp earlier than it | `fault` "no heartbeat since the last start" (`I-no-poll-after-restart`) |
| `head`: the tree's HEAD commit time more than 10 minutes after `ActiveEnterTimestamp` | `fault` "scheduler predates the deployed tree" (`I-stale-deploy`) |
| otherwise | `verified`, with the evidence recorded |

**Notes:**
- **Each row is its own result,** with its own incident. One unavailable input never hides the others.
- **What counts as an instance:** a process whose argv runs the `bts` entry point with the `schedule` subcommand, **and whose parent process is not itself such a process**.
  - `uv run bts schedule` starts a `uv` parent and a Python child, and both argvs contain `bts schedule`. Counting raw matches would report every healthy scheduler as two.
  - **Observed on the box 2026-10-06** (read-only `systemctl --user cat` and `ps`):
    - the unit is `ExecStart=/home/bts/.local/bin/uv run bts schedule --config /home/bts/.bts-orchestrator.toml`, with `Restart=always`;
    - the live tree is a `uv run bts schedule` parent and its child `…/.venv/bin/python3 …/.venv/bin/bts schedule`, whose parent PID is the `uv` process.
    - The unit file is not tracked in the repository. The fixtures mirror this tree, and the tree rule counts it as one instance.
- **The watchdog's churn samples** are its own (owned root), so it never writes the monitor's state. The monitor's file is only read for its freshness.
- **The deploy-version row (reasoning only):**
  - the deploy workflow runs `git pull` and then restarts both units within seconds, so a HEAD newer than the unit's start by more than 10 minutes means a restart did not happen;
  - a HEAD **older** than the start is normal (a scheduled daily restart);
  - the heartbeat persists no code version, so this is the strongest observation available.
- **No-games days:** the scheduler sleeps; `is_stale` already handles `sleeping` with `sleeping_until`.

## 6. W7: W-self
**Registration row:**
- a dedicated Healthchecks check, commissioned by Eric, observes completion of due watchdog work;
- it is independent of scheduler health and of the generic cron ping;
- per-job completion, lock skips, overdue work, checker exceptions and pending alert delivery are recorded separately;
- a standalone unconditional ping cannot establish completion.

**C2: three parts:**
1. **Per-job completion records:** `data/watchdog/self/<job>.json`, written by `bts watchdog run` after every attempt. It holds:
   - `started_at`, `ended_at`, `elapsed_s` and `budget_exhausted`;
   - `outcome`: `completed`, `infrastructure_error`, `lock_skipped` or `registration_error`;
   - checker-failure count and business-alert count;
   - pending-notice count and the oldest pending age;
   - the ping result.

   A lock skip (`JobBusy`) is recorded as `lock_skipped`, never as completion.
2. **The `self` check** in the `fast` job. It reads every registered job's record and compares it with that job's declared period:
   - `fast` 5 minutes;
   - later units add theirs.

   | Condition | Result |
   |---|---|
   | a job's last `completed` is older than its period + grace (5 minutes for `fast`) | `fault` "overdue" (`I-overdue-<job>`) |
   | three or more consecutive `lock_skipped` | `fault` "lock starvation" (`I-lock-starved-<job>`) |
   | a notice pending for more than 30 minutes | `fault` "alert delivery pending" (`I-alert-pending`) |

   The fault notice itself may not be deliverable. That is why part 3 exists.
3. **The dedicated ping:**
   - **The URL:** `BTS_WATCHDOG_PING_URL` from the environment. The name is fixed here, the value is never committed, and Eric commissions the check.
   - **When the job ends,** the CLI pings `<url>` only when all of these hold:
     - the outcome is `completed`;
     - results persisted and notification ran without an infrastructure error;
     - no checker failure;
     - no notice pending for more than 30 minutes.

     Otherwise it pings `<url>/fail`.
   - **Failures:** a missing URL, or a failed ping, is recorded and never raises. The run's exit code is unchanged by the ping.
   - **What Healthchecks then sees:** a missing or failed ping after its period plus grace. That covers a dead cron, a hung job, starvation, checker exceptions and an undeliverable DM. None of these depends on the DM path working.
   - **Commissioning (operational acceptance, registration §4.3, not this unit's code gate):**
     - Eric creates the check (period 5 minutes, grace 10 minutes);
     - the URL goes into the box `.env`;
     - a missed-completion rehearsal is observed.

## 7. The C1 draft's open questions, resolved
1. **The 5-minute decision grace:** kept.
   - **Why (reasoning only):** the decision is written after the live-forward capture subprocess (10 s timeout) and the send.
   - **The trade:** a late write shows as `pending` for up to 5 minutes; the alternative is a false fault.
2. **Pre-cutoff risk:** only the heartbeat's persisted `sleeping_until` is used. Planned check times are not persisted (map gap 1), and the spec does not reconstruct them.
3. **A candidate on a postponed game:** `pending` until its cutoff (§2).
4. **The carry horizon:** 3 days, plus yesterday always. The unresolved-disposition deadline is 06:15 ET the next day.
5. **`crontab -l`:** yes, through the process seam (§0.4).

## 8. Tests
**Gate 1 shape** (registration §4.1): red then green fixtures drive the **real** `bts watchdog run fast` CLI. Only the leaves are patched:
- `proc.run`;
- the MLB fetch;
- the clock (wall and monotonic);
- the DM transport;
- the Healthchecks ping.

The predicate under test is never patched.

**Coverage per predicate:**
- every predicate has a fault with its incident and selection, a no-fault control, an unavailable-input case, and a fault → repair → recovered → recurrence sequence;
- **cutoff edges:** equality at the cutoff, an earlier double-down leg, a refreshed time that differs from the pick, UTC/ET date disagreement, midnight carry, DST days, early and late games;
- the draft's §5 W-deliver, W-postponed, W-decision and W-mode cases;
- **W-restart:** stale heartbeat; churn at each threshold; NRestarts unavailable; monitor missing; monitor silent; two schedulers; an inactive unit; no poll after a restart; a stale deploy; a no-games sleeping day;
- **W-self:**
  - an overdue job, lock starvation and an old pending notice;
  - a success ping only on a clean completion;
  - `/fail` on a checker failure, on an infrastructure error, and on a notice pending past 30 minutes;
  - a missing URL recorded;
  - a ping failure that does not change the exit code;
- **the job budget:** an exhausted budget gives `unverifiable` results, a bounded flush, and `budget_exhausted` recorded.

**Producer-to-reader fixtures (C1 lesson):**
- pick files, decisions, scheduler states and archives are produced by running the **real** writers (`save_pick`, `write_decision`, the scheduler's state saver, `_archive_and_remove_pick`) into a temporary data directory;
- hand-written JSON is used only for corrupt and legacy cases.

**Gate 2 (W0's kernel gate):** the `fast` job runs with fixtures inside the sandbox, with the seams substituted.
- It writes only beneath `data/watchdog`, and it creates no process.
- **A red control:** a mutant that calls the real `proc.run` inside the gate must kill the gate.

**Mutation ledger:** pinned before review, at least one mutant per predicate row and per budget or ping rule.

## 9. What is not claimed
- W-postponed reports current status; it never reports the status at send time.
- W-restart does not replace the external monitor. It declares it a dependency and checks it independently.
- W-self's dedicated check is operational only after Eric commissions it. Until then, watchdog death is not independently monitored, and that is said in every status.
- Nothing here changes a pick, a decision, the streak, the scheduler or the cron (R1).

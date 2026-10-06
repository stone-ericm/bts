# C2 (a) unit 2 spec: the five-minute job (W1, W5, W7), revision d2

**What this is:** the spec for C2 item (a), review unit 2 (C2 proposal §5 (a)). It covers these registration rows of `docs/sota_audit/2026-10-04-prereg-c1-watchdog.md` (FROZEN):
- **W1:** W-deliver, W-postponed, W-decision and W-mode;
- **W5:** W-restart;
- **W7:** W-self;
- the §3 schedule and execution bound for the five-minute job.

**Status:** revision **d2**, for spec review round 2 of 2 (the last). It answers design review d1 (`docs/audit/2026-10-06-c2-unit2-spec-codex-d1.md`, REVISE, findings 1–12). Production code: reviewed until SIGN, deployed only with the whole watchdog under Eric's D7.

**Built on:** W0, signed in C2 unit 1 (`f1b948a`, branch `c2-watchdog`). Supersedes the unreviewed C1 W1 draft (`docs/superpowers/specs/2026-10-06-c1-r2-w1-spec.md`) and its evidence map, and revision d1 of this spec (`b12c018`).

**Not in this unit:** W2 (entry), W4 (grade), W6 (restore), W8 (capture-stop with rank-1 Part A) and W9 (the coverage matrix) are units 3–4. W3 (W-state) stays deferred with P3. The cron line and deployment come with the whole watchdog's D7.

## Changes from d1, by finding
| d1 finding | Where this revision answers it |
|---|---|
| 1. the budget is not an execution bound | §0.2: a signal-enforced hard deadline over the whole run, phase limits, capped network bodies, a bounded finalizer |
| 2. W-self success and record schema | §6: a static expected-work inventory, a qualified completion rule, a durable attempt record with last success and a skip streak, and success blocked by any W-self alert |
| 3. finalization gaps | §3: an unrecorded skip at the completion deadline; attempts aged from the watchdog's own first observation; a lock without a decision; scoring results limited to hit/miss |
| 4. mixed-file verdicts | §0.4: verdict classes. Cross-file verdicts are pending before the date's completion deadline, become an "overdue completion" fault after it, and are verified only after it. Unreadable archives are pending |
| 5. date and selection inventory | §0.3: the watchdog's own day inventory from install, with no age cutoff and independent of notification state; first-observation timers; historical incidents never auto-recover; current conditions keep their episode date |
| 6. W-deliver eligibility, receipts, warnings, freshness | §1 and §0.6: eligible opportunities via the production void classifier and `officialDate`; receipt timing without the decision; the warning only inside the 75-minute entry window with a fresh heartbeat; scheduled-check times dispositioned as deferred (§7); cached schedule observations labelled with their fetch time and refreshed every run for active dates |
| 7. W-restart deploy and polling | §5 and §7: the deploy-version and polling-progress rows are **deferred** with reasons; the row that remains is "no heartbeat since the last start", named as what it is; `head` is removed from the seam; each input has its own unverifiable result |
| 8. external monitor | §5: the monitor's cron line is parsed for cadence and arguments, its sample coverage is checked (not file mtime), and the watchdog's own churn baseline is qualified |
| 9. W-mode | §4: an intent epoch from the watchdog's own continuous observation, with earlier evidence unverifiable; installed-cron parsing without relying on the marker; unsupported forms alert |
| 10. parser qualification | §0.5: validators after every compatibility parser; unavailable input takes precedence over no-games, skip and all-clear |
| 11. the DM credential subprocess | §0.6: a watchdog transport with environment-only credentials and single attempts; network-leaf contracts; real-seam tests |
| 12. the gate plan | §8: named fixtures for every counterexample, producer-pause tests, a cron-selection gate, confinement branches, and a mutation plan by failure mechanism |

## 0. Common rules

### 0.1 The job and its expected work
- **Registration:** `register("fast", [("deliver", …), ("postponed", …), ("decision", …), ("mode", …), ("restart", …), ("self", …)])`. The business check names are `W-deliver`, `W-postponed`, `W-decision`, `W-mode`, `W-restart` and `W-self`.
- **Schedule (R3, §3):** every five minutes, every day and hour, games or not.
- **The expected-work inventory** is `EXPECTED` in `bts/watchdog/expected.py`: a static, reviewed table of every job, its check ids, its period and its grace. For this unit it is `fast`: checks as above, period 300 s, grace 300 s.
  - The registry (`cli.JOBS`) is compared with it on every run.
  - **Registry drift** (a job or check missing, extra or renamed) is a W-self fault, and blocks the success ping (§6). A check removed from the code therefore cannot silently stop being due.

### 0.2 The hard deadline (registration §3, "a two-minute execution bound")
Everything is timed from **admission**, the moment `bts watchdog run fast` holds its job lock (W0's lock is non-blocking). All times use the monotonic clock.

| Phase | Rule |
|---|---|
| Predicates | A check starts only while elapsed < 60 s. A check not started gives one `unverifiable` result ("deadline: not run"), and the attempt is `incomplete` (§6) |
| Seam calls | timeout = `min(cap, 100 s − elapsed)`. A call that would get < 2 s is not made, and its predicates are `unverifiable` |
| Notification flush | W0's monotonic budget = `min(90 s, 100 s − elapsed)`, so the 90 s notifier cap is never raised |
| **Hard stop** | at 105 s, `signal.setitimer(ITIMER_REAL)` fires in the main thread. Its handler raises `JobDeadline` (a `BaseException`, so a check's `except Exception` cannot swallow it). It interrupts any blocking call (socket, `flock`, sleep): Python retries a syscall after EINTR only when the handler does not raise. The attempt is `deadline_exceeded` |
| Finalizer | runs after the job, or after `JobDeadline`, under its own 10 s alarm. It writes the attempt record (§6), then pings (timeout ≤ 5 s, body ≤ 1 KiB), then records the ping result |

- **The worst case** is 105 + 10 s plus interpreter start, inside 120 s. If the finalizer itself overruns, the CLI exits non-zero with no ping; the missing ping is what the dedicated check detects (§6).
- **An interrupted send:** a notice whose send was in flight at the hard stop keeps its W0 claim. After the lease it is retried, so a duplicate is possible; that is W0's documented limit, and it is never lost.
- **Atomic writes:** an interrupted `write_atomic` removes its own temp file (W0 `except BaseException`).
- **Tests** drive the deadline through an injectable timer. One test also uses a real 0.5 s alarm against a sender that blocks for 5 s.

### 0.3 The day inventory and first observations (all watchdog-owned)
- **`fast/days.json`:** `{version, inventory_start, days: {D: {status: open|closed, closed_at, basis}}}`.
  - **`inventory_start`** is the first run's ET date.
  - **Every run** first adds every missing date from `inventory_start` through today as `open`, so days missed during watchdog downtime are not skipped.
  - **Every `open` date is evaluated on every run**, with no age cutoff.
  - **A date closes** when, **after its completion deadline** (§0.4), every result of its date-scoped checks (`deliver`, `postponed`, `decision`) is **final**:
    - `verified`;
    - an H-class `fault` (a dated event: it stays a fact, and its target stays open for the operator);
    - a W-postponed `fault` (after the deadline the slot's void or missing status is settled).

    "No eligible games" verified also closes the date. An X-class "overdue completion" fault is **not** final: the date stays open and is re-evaluated, so a later repair verifies and recovers it. A closed date is no longer evaluated, and its H targets stay open in the notification state.
  - **An open date whose required inputs are still unavailable 24 h after its completion deadline** turns those results into a fault, "unresolved date: inputs unavailable" (`I-date-inputs-unavailable`). It is never silently retired.
- **`fast/seen.json`:** the first-observation time of each pending condition, keyed by (date, check, condition, selection). Overdue timers (§0.4) run from these times, which survive scheduler restarts and are never reset by them. A condition that stops being observed is removed.
- **Neither file depends on the notification state.**
- **Corruption:** a corrupt or unreadable inventory or seen-file is copied aside within the owned root (`<name>.corrupt-<attempt_id>`) and rebuilt.
  - **The rebuilt inventory** opens the last 7 dates, and the run reports a fault (`I-inventory-corrupt` / `I-seen-corrupt`).
  - **The attempt is `incomplete`.**
  - **Rebuilt timers restart,** which can only delay an overdue alert. This limit is stated.

### 0.4 Verdict classes
Each result row in §§1–5 is tagged with one class.

| Class | Meaning | `verified` | `fault` |
|---|---|---|---|
| **S** (single source) | the verdict reads one file, or one file plus the schedule | immediately | immediately, or after **two consecutive runs** observing it where marked "persistent" (registration: "a persistent overdue …") |
| **X** (cross-file) | the verdict compares two or more files written at different times | only **after** the date's completion deadline; before it, agreement is `pending` ("not final before <deadline>") | before the deadline, disagreement or absence is `pending`, with a first-observation time. It becomes a fault "overdue completion: <condition> since <first seen>" either when the condition's own timer expires or at the deadline. **A fault never asserts corruption, only overdue completion** |
| **C** (current condition) | a present-tense fact about the system (liveness, config, cron) | when the condition clears | while it holds |
| **H** (historical fact) | a dated event that happened (a late send, a refusal, a missed early game) | never automatically: the target stays open | once observed |

**Completion deadlines:**
- 06:15 ET on D+1 for the scheduler's own records (pick, decision, state, archives). The end-of-day skip can land after midnight.
- 08:15 ET on D+1 for anything involving results. The 01:00 grader can run to 06:00, and reconcile runs at 07:40 under its 08:00 cutoff.

**Two consequences:**
- **Mixed reads:** a mixed read mid-transaction can only produce `pending`. It never produces a fault or a recovery.
- **Recovery:** a C-class result keeps the date of the open episode it would recover, so a condition opened before midnight recovers on the same target. That date is read from the notification state; if the state is unreadable, the result uses today. H-class targets stay open: a changed current candidate never manufactures an all-clear.

### 0.5 Input qualification
Each input is read once per run (`read_bytes`), parsed with the production compatibility parser where one exists, then **validated**. A failed validation makes the input **unavailable**. An unavailable input makes each predicate needing it `unverifiable` (or `pending` in class X). That **takes precedence** over no-games, skip and all-clear outcomes.

| Input | Validator (after parsing) |
|---|---|
| pick `data/picks/<D>.json` | `parse_daily_pick`, then: `date == D`; `pick` and any `double_down` have positive int `batter_id`/`game_pk` and a parseable `game_time` (naive = UTC); the booleans `bluesky_posted`, `notification_sent` and `delivery_attempted` are exactly `bool`; `bluesky_uri` and `notification_id` are a non-empty `str` or `None`; `delivered_at` is `None` or tz-aware ISO; `result` ∈ {hit, miss, void, suspended, unresolved, None}; `slot_results` keys ⊆ {pick, double_down} with values ∈ {hit, miss, void} |
| **qualified receipt** (derived) | either `notification_sent is True`, `notification_channel == "bluesky_dm"`, a non-empty `notification_id` and a `delivered_at`; or `bluesky_posted is True`, a non-empty `bluesky_uri` and a `delivered_at`. `pick_was_delivered` being true without a qualified receipt is **unavailable** ("unqualified receipt") |
| decision `data/picks/<D>/decision.json` | `parse_decision` non-`None`, then: `date == D` as a string; `source` ∈ {mdp, heuristic, unknown}; `delivery_status` ∈ the four values; `finalized_at` parseable; `primary` `None` or `{batter_id: int, game_pk: int, …}`; `double_down` likewise. **A structural failure is unavailable.** Two **semantic** inconsistencies on a structurally valid record are faults, not unavailable (§3): `skip` with `scoreable == true`, and a commit (`single`/`double`) with `scoreable == false` |
| scheduler state | a JSON object; each field a predicate uses has its production type (`final_skip_candidate` null/object, `committed_pick_written` bool, `pick_locked` bool, `delivery_refusals` list of objects with a parseable `cutoff_et`). A missing field that is used is unavailable, never an empty default |
| archives `<D>/{stale_pick,deferred_fallback,refused_delivery}_*.json` | a JSON object with the pick fields, plus its envelope key (`stale_pick`/`deferred_fallback`/`refused_delivery`) holding `reason` and its time key. An unreadable or invalid archive (it may be mid-`write_text`) is `pending`; after 15 minutes from first observation it is a fault (`I-archive-unreadable`) |
| schedule (MLB) | an object with a `dates` list; each game has an int `gamePk`, a parseable `gameDate`, a string `status.detailedState`, `officialDate` `YYYY-MM-DD` and `gameType`. A malformed response is unavailable, never "no games" |
| heartbeat `data/.heartbeat` | an object; `state` ∈ the four states; a tz-aware `timestamp`; a tz-aware `sleeping_until` when `state == sleeping` |

### 0.6 Seams and network leaves (R6)
**Process seam `bts.watchdog.proc`:** `run(name, timeout)` for exactly four commands; any other name raises.

| Name | argv |
|---|---|
| `crontab` | `crontab -l` |
| `nrestarts` | `systemctl --user show bts-scheduler -p NRestarts --value` (`read_nrestarts`' exact argv, through an adapter that accepts nothing else) |
| `unit` | `systemctl --user show bts-scheduler -p ActiveState,SubState,MainPID,ActiveEnterTimestamp` |
| `ps` | `ps -eo pid=,ppid=,args=` |

- **How it runs:**
  - no shell; `stdin=DEVNULL`;
  - a minimal environment: `PATH=/usr/bin:/bin`, `HOME`, `LANG=C`, plus `read_nrestarts`' `XDG_RUNTIME_DIR` / `DBUS_SESSION_BUS_ADDRESS` defaults;
  - stdout and stderr are each capped at 64 KiB, and a capped read is an error;
  - a timeout kills and reaps the child (`subprocess.run`);
  - a timeout, a non-zero exit or an error is returned as data, never raised past the predicate.
- **In the gates:** gates 1–2 substitute `run`. Inside gate 2 any other process creation still kills the gate (unit 1).
- **Direct tests of the real seam,** outside the gate:
  - exact argv and environment;
  - unknown names refuse;
  - stdout and stderr caps;
  - timeout kill and reap;
  - non-zero and error shapes.
- **Stated boundary:** the four real children are system binaries run read-only, outside the kernel write gate.

**MLB schedule:**
- **The request:** `GET https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=<D>` with `urllib`. One attempt, the §0.2 timeout, the body capped at 2 MiB.
- **Today and every open date with an S- or C-class current-status condition are refreshed every run.** Other open dates use a cached response up to 30 minutes old (`data/watchdog/cache/schedule-<D>.json`).
- **Every status-based result names the observation time** of the response it used ("status as of <fetched_at>"). A cached response is never labelled with this run's time.
- **No production helper that fetches the schedule itself is called.**

**DM transport `bts.watchdog.transport`** (closes d1 finding 11):
- **The credential comes from the environment only,** in `bts.posting.get_bluesky_password`'s order of environment names (`BTS_BLUESKY_APP_PASSWORD`, `BTS_BLUESKY_PASSWORD`, `BTS_BLUESKY_DM_PASSWORD`). The keychain fallback and its `security` subprocesses are never reached.
- **A missing credential** raises `TransportUnavailable`. The notice stays pending (W0), and W-self's pending-alert rule (§6) applies.
- **The calls:** the four AT-protocol calls of `bts.dm.send_dm` (create session, resolve handle, get conversation, send). They are mirrored locally, so production `bts.dm` is unchanged and no re-certification is triggered.
- **Each call:** one attempt, no retry, timeout `min(10 s, remaining)`, the response body capped at 64 KiB.
- **The message id** is returned only from a parsed send response with a non-empty string `id`.

**Dedicated ping:**
- `GET <BTS_WATCHDOG_PING_URL>` or `…/fail`, with timeout `min(5 s, remaining)` and the body read capped at 1 KiB.
- It never raises. The URL value is never logged or committed.

### 0.7 Times, statuses, selections, eligibility
- **Clocks:** ET-aware via the W0 clock seam. A pick's `game_time` is UTC, and a naive value is read as UTC.
- **Statuses:** W0's. **Selections:** `<batter_id>@<game_pk>`, legs joined with `+`, or `day`.
- **An eligible game for D** passes four tests:
  - `bts.util.is_regular_season_game`;
  - `officialDate == D` (a resumed suspended game is excluded);
  - `bts.picks._is_void_detailed_state` is false on its `detailedState` (postponed and cancelled are excluded);
  - a parseable `gameDate`.
- **The day's opportunity set** is its eligible games. **No eligible games** is `verified` "no eligible games" and closes the date. The current postseason is not regular season, so the box's idle days verify this way.
- **A leg's refreshed first pitch** is its game's `gameDate` from the schedule.
- **The cutoff** of a selection is the minimum over its legs of (refreshed first pitch − `SUBMISSION_CUTOFF_MIN`). Equality is past. A leg absent from the schedule has no refreshed time, and is W-postponed evidence.

### 0.8 Mode gating and the intent epoch (R4)
- **The intent** is `check_entry_intent(tomllib.load(--config))`.
- **`mode/intent.json`** (owned) holds `{intent, delivery, since}`. `since` is the start of the current run when this run observes a valid (intent, delivery) pair different from the stored one, or when the previous run could not read it.
- **The epoch is the watchdog's own continuous observation.** No production intent history exists, and config mtime is not one.
- **Delivery evidence** after `since + 5 min` is attributable to the stored intent. Earlier evidence is `unverifiable` ("intent at that time not observed").
- **W-deliver's delivery expectations apply only under `enter`.** Under `research` they are `verified` ("research: no delivery expected"). Liveness, mode, finalization and self checks run under every intent. An intent problem makes the delivery expectations `unverifiable`, and W-mode reports the problem.

## 1. W-deliver (under `enter`; per open date D)
| Row | Class | Evidence → result |
|---|---|---|
| D1 late receipt | H | a qualified receipt (independent of any decision) with `delivered_at` ≥ the cutoff of its selection → `fault` `I-late-delivery`; strictly before → `verified` |
| D2 refusal | H | a valid `refused_delivery_*` archive → `fault` `I-delivery-refused` (one per archive) |
| D3 overdue live candidate | S, persistent | a valid pick with no qualified receipt and `delivery_attempted == false`, past its cutoff, the same selection in two consecutive runs, and no suppressor read in the same run → `fault` `I-overdue-candidate`. Suppressors: the state's `final_skip_candidate` set with `committed_pick_written == false`, a valid skip decision, or a deferral archive for that selection |
| D4 attempt unresolved | S, timer | `delivery_attempted == true` without a qualified receipt: `pending`, then `fault` `I-attempt-unresolved` 15 minutes after first observation. It recovers when the pick shows a qualified receipt or `delivery_attempted == false` |
| D5 wake after cutoff | S | a live candidate without a receipt; now within [its first pitch − 75 min, its cutoff); the heartbeat valid and **not stale** (`is_stale`); `state == sleeping` with `sleeping_until` ≥ the cutoff → `fault` `I-wake-after-cutoff`. A stale or invalid heartbeat → `unverifiable`. Outside the window: no result |
| D6 no candidate at all | S, persistent | after the last eligible game's cutoff: no pick, no archive and no skip evidence for D, in two consecutive runs → `fault` `I-no-disposition` |
| D7 missed early game | X (06:15) | a `stale_pick_*` archive (a candidate that passed its cutoff undelivered), unless D's final decision is an MDP skip → `fault` `I-missed-early-game`. A later game that was then delivered is reported by D1 and is not this fault |

- **The 75-minute window** is `check-pick-entered`'s default `--window-min` (`cli.py`), the registration's "existing entry window".
- **Planned check times are not persisted.** The scheduled-check half of the registration's pre-cutoff rule is deferred (§7).
- **Text:** post-cutoff faults never say "enter now".

## 2. W-postponed (per open date D, both legs)
**The committed selection** comes from one file: the pick with a qualified receipt, or else a valid commit decision's `primary` (plus its `double_down` only when the action is `double`).

| Row | Class | Evidence → result |
|---|---|---|
| P1 committed slot void | C | a committed leg whose current status is void, or whose `officialDate` ≠ D → `fault` `I-postponed-slot` ("status as of <fetched_at>") |
| P2 committed slot missing | C | a committed leg absent from D's schedule → `fault` `I-game-missing` |
| P3 live candidate void or missing | S | `pending` until its cutoff ("the scheduler normally re-selects"); at the cutoff, D3 or D6 apply |
| otherwise | C | `verified`, as of the response time |

Results report current status only, never the status at send time. Suspension after play is W-grade's.

## 3. W-decision (finalization; per open date D; every intent)
| Row | Class | Evidence → result |
|---|---|---|
| F1 receipt without decision | X (timer 15 min, deadline 06:15) | a qualified receipt and no valid commit decision → `pending`, then `fault` `I-decision-missing` ("overdue completion") |
| F2 lock without decision | X (06:15) | the state's `pick_locked == true` and no valid decision → `pending`, then at the deadline `fault` `I-lock-without-decision`. This covers a private commit that never published, and a classification lock, which writes no decision. `pick_locked` alone never verifies anything |
| F3 unrecorded skip | X (06:15) | the state's `final_skip_candidate` set, `committed_pick_written == false`, and no valid decision at the deadline → `fault` `I-skip-unrecorded` |
| F4 unresolved day | X (06:15) | eligible games existed and, at the deadline, there is neither a valid commit decision nor a valid skip decision (and F2/F3 do not apply) → `fault` `I-unresolved-disposition` |
| F5 inconsistent record | S | a structurally valid decision with `skip` + `scoreable == true`, or a commit with `scoreable == false` → `fault` `I-decision-inconsistent` |
| F6 selection mismatch | X (06:15) | a valid commit decision against the committed pick: `primary` ≠ the pick's leg; `double` without the pick's `double_down`, or with a different one; `single` while the pick file has a `double_down` leg → `pending`, then `fault` `I-selection-mismatch`. On a `single` record the decision's `double_down` (the runner-up) is not compared |
| F7 score-bearing preview | X (08:15) | the pick's `result` or a `slot_results` value ∈ {hit, miss}, with no qualified receipt and no valid commit decision → `fault` `I-score-bearing-preview`. `void`, `suspended` and `unresolved` never count as scoring |
| F8 needed pick absent | X | a valid commit decision with no pick file → `pending`, then at the deadline `fault` `I-committed-pick-missing` |
| otherwise | X | `verified` after the deadline |

## 4. W-mode (current conditions; every run)
| Row | Class | Evidence → result |
|---|---|---|
| M1 configuration | C | `--config` unreadable or unparseable, or `check_entry_intent` problems → `fault` `I-config-intent` |
| M2 entry cron | C | below → `enter` without an active checker: `fault` `I-enter-without-checker`; `research` with one: `fault` `I-research-with-checker`; an unsupported form: `fault` `I-cron-unsupported`; `crontab` unavailable: `unverifiable` |
| M3 delivery under research | C/H | a qualified receipt with `delivered_at` > `since + 5 min` while the stored intent is `research` → `fault` `I-delivered-in-research`. Evidence before `since` is `unverifiable` |
| M4 private under enter | C/H | a valid `private_locked` decision with `finalized_at` > `since + 5 min` while the stored intent is `enter` → `fault` `I-private-under-enter` |

**Crontab parsing (M2 and §5):**
- Blank lines, `#` comment lines and `NAME=value` lines are skipped. Every other line must be 5 time fields plus a command, or it is "unsupported".
- **An active entry checker** is a line whose command contains, as whitespace-separated tokens, `run bts check-pick-entered` after a `uv` executable (the installer's form). The `# BTS-HETZNER` marker is not required.
- **Any other uncommented line containing `check-pick-entered`** is an unsupported form: `fault`, since it may be an active checker.

## 5. W-restart (current conditions; date of the open episode, else today)
**Reuse:** `is_stale`, `assess_churn` and `read_nrestarts` from `scripts/check_heartbeat.py`, loaded by file path (`importlib.util.spec_from_file_location`) from the repository root. Its `main()` is never called.
- **`is_stale` reads the heartbeat itself.** That read is used only for R1's staleness verdict. D5 and R7 use the watchdog's own validated read, and no verdict compares the two reads.

| Row | Evidence → result |
|---|---|
| R1 heartbeat stale | `is_stale(path, now)` stale → `fault` `I-heartbeat-stale`, else `verified` |
| R2 restart churn | `nrestarts` through the adapter, then `assess_churn(own samples, n, now)` with the samples at `restart/churn.json`. Churn → `fault` `I-restart-churn`. A counter that is unavailable → `unverifiable`. **Baseline coverage:** while the oldest kept sample is younger than 180 min, a no-churn return is `pending` ("baseline <age> of 180 min"), not `verified`; this covers first install, a counter reset (the helper prunes larger samples) and a gap. A corrupt own-history file is set aside and reset, with a one-run `fault` `I-churn-history-corrupt` |
| R3 monitor configured | the crontab's heartbeat line. Exactly one uncommented line whose time fields are `*/5 * * * *` and whose command runs `scripts/check_heartbeat.py` with `--heartbeat-path data/.heartbeat`, a `--ping-url` with a non-empty argument (its value not read), no `--no-churn`, `--churn-unit` absent or `bts-scheduler`, and no `--churn-state` → `verified`. No such line → `fault` `I-monitor-missing`. A line that differs → `fault` `I-monitor-misconfigured` |
| R4 monitor observing | only when R1 is fresh: the monitor's `data/health_state/scheduler_churn.json` (read only) is an object with `unit == "bts-scheduler"` and a latest sample within 20 minutes → `verified`. Empty samples (the monitor's counter was unavailable) or an older latest sample → `fault` `I-monitor-counter-unavailable`. This attests only the monitor's local observation; its external ping is outside the watchdog's view |
| R5 instances | `ps`: a **scheduler process** is one whose argv tokens are `[…/uv, run, bts, schedule, …]` or `[<python>, …/bin/bts, schedule, …]`. An **instance** is a scheduler process whose parent is not one. More than one → `fault` `I-multiple-schedulers`. None while R6 says active → `fault` `I-scheduler-process-missing`. Output that cannot be parsed → `unverifiable`. **Observed on the box 2026-10-06** (read-only `systemctl --user cat`, `ps`): `ExecStart=/home/bts/.local/bin/uv run bts schedule --config /home/bts/.bts-orchestrator.toml`; the live tree is the `uv run bts schedule` parent plus its child `…/.venv/bin/python3 …/.venv/bin/bts schedule`, which counts as one |
| R6 unit active | `unit` `ActiveState != active` → `fault` `I-scheduler-inactive`. Unparseable → `unverifiable` |
| R7 heartbeat since start | `ActiveEnterTimestamp` more than 10 minutes ago, and the watchdog's validated heartbeat `timestamp` earlier than it → `fault` `I-no-heartbeat-since-start`. This is a liveness fact, not a claim about polling. Either input unavailable → `unverifiable`. **Why a healthy start passes:** `run_day` writes a RUNNING heartbeat right after state init and before any wait or schedule fetch (`scheduler.py`, the `write_heartbeat` before "1. Fetch schedule"), so a healthy start writes one within seconds of `ActiveEnterTimestamp` |

## 6. W-self
**The attempt record** `self/<job>.json` is updated read-modify-write under the owned lock `self/.lock`, which is distinct from the job lock:

```
{version: 1, job,
 last_attempt: {attempt_id, started_at, ended_at, outcome, elapsed_s, deadline_hit,
                checks_completed: [...], checker_failures, alerts, pending_oldest_s, ping},
 last_completed_at, last_completed_attempt_id, consecutive_lock_skips,
 recent: [the last 20 attempt summaries]}
```

- **`outcome`** is one of:
  - **`completed`:** every expected check ran to valid results (`executed_ok`), the results persisted, notification raised no infrastructure error, and no deadline was hit;
  - `incomplete`; `infrastructure_error`; `deadline_exceeded`; `lock_skipped`; `registration_error`.
- **A lock skip** (`JobBusy`) increments `consecutive_lock_skips`; any other ended attempt resets it.
- **Every attempt has a uuid.** Because updates serialize on `self/.lock`, a skip racing a running attempt's completion loses neither update. `last_attempt` is the attempt that ended most recently.
- **First run:** no record, so one is created.
- **A corrupt record** is set aside within the owned root and recreated. The self check reports `I-self-record-corrupt` once, and that attempt is `incomplete`.

**The `self` check** (runs inside `fast`; reads every expected job's record and the notification state):

| Row | Evidence → result |
|---|---|
| E1 overdue | now − `last_completed_at` > period + grace → `fault` `I-overdue-<job>`. No record yet → `pending` ("no completion yet") |
| E2 lock starvation | `consecutive_lock_skips` ≥ 3 → `fault` `I-lock-starved-<job>` |
| E3 alert pending | the oldest pending notice older than 30 minutes → `fault` `I-alert-pending` |
| E4 registry drift | `cli.JOBS` ≠ `EXPECTED` → `fault` `I-registry-drift` |
| E5 inventory health | a corrupt inventory, seen-file or self record this run → `fault` (named in §0.3 and above) |

**The success ping:** `<url>` only when all of these hold:
1. this attempt is `completed`;
2. the `self` check produced **no alerting result**, whether or not its DM was delivered;
3. the attempt record persisted.

Otherwise `<url>/fail`.
- **Other checks' business faults** (a late pick, say) **do not block success:** the watchdog completed its work and delivered the alert.
- **The order:** persist the record, then ping, then record the ping result (best effort).

**Detection bounds:**
- **The local overdue rule** (5 + 5 minutes) is detected by the next `fast` run that executes.
- **The dedicated check** (period 5 minutes, grace 10 minutes, commissioned by Eric) detects anything that stops success pings within about 15 minutes. That includes `fast` itself dying, which no local check can observe.
- **Commissioning** is Eric's: the check, the URL in the box `.env`, and a missed-completion rehearsal. These are operational acceptance (registration §4.3), not this code gate.
- **Until commissioned,** watchdog death is not independently monitored, and the status says so.

## 7. Explicit dispositions (recorded in W9's coverage matrix)
| Registration element | Disposition and reason |
|---|---|
| W-deliver: "a scheduled check/wake that cannot precede the candidate cutoff" (the scheduled-check half) | **Deferred.** Planned check times are not persisted (W1 evidence map, gap 1). It needs a producer prerequisite (the scheduler persisting its planned run times), which is not in C2. The current-wake half is D5 |
| W-restart: deploy-version observations | **Deferred.** The running scheduler persists no loaded revision, and commit time is not install or load time (d1 finding 7). It needs a producer prerequisite (the scheduler stamping its code revision at start) |
| W-restart: absent polling after restart | **Narrowed to R7** (no heartbeat since the last start). Result-polling progress needs a producer receipt of poll progress; deferred |
| Multi-file closure (registration §3 consistent capture) | **Not claimed.** Cross-file verdicts are class X (pending, then overdue completion); P3/W-state stays deferred (row C1-r2-p3-deferral) |
| Historical mode attribution before the watchdog observed the intent | **Unverifiable** (§0.8) |

## 8. Tests (registration §4 gates 1–3)
**Fixture rules:**
- Fixtures run the **real** producers into a temporary data directory: `save_pick`, `write_decision`, `save_state`, `_archive_and_remove_pick`, and the scheduler's `_deliver_and_lock_pick` and `_write_endofday_skip`, with only their transport, contest-state, capture and clock leaves patched. Hand-written JSON only for corrupt, legacy and wrong-type cases.
- Every case drives the **real** `bts watchdog run fast` CLI. Only `proc.run`, the MLB fetch, the clocks (wall, monotonic and the deadline timer), the DM transport's HTTP leaf and the ping leaf are substituted. The predicates are never substituted.

| Group | Named cases (each red first, then green) |
|---|---|
| Deadline | hung sender (fake timer and one real 0.5 s alarm); blocked state lock; predicates past 60 s; flush cap at 90 s; finalizer persists `deadline_exceeded` and pings `/fail`; an interrupted claimed notice is retried after its lease |
| W-self | budget-exhausted placeholders give no success ping; a check removed from the registry; a job missing; three lock skips; a skip racing a completion (threads); a missing record; a corrupt record; a record write failure; a completed run with a delivered business fault (success ping with a confirmed fake id); W-self's own fault delivered and still `/fail`; a missing ping URL recorded |
| Finalization | the real skip writer failing after the pending skip is saved, then crossing 06:15; the real send paused after `delivery_attempted` is saved, and a restart not resetting the 15-minute timer; a private commit paused before its decision; `suspended`/`unresolved` results on a preview; `single` against a pick with a DD leg; `double` against a pick without one |
| Capture | the real scheduler paused between pick, decision and state saves while the watchdog runs: `pending`, with no fault and no recovery; then the completed state after the deadline is `verified` |
| Dates and recovery | a multi-day watchdog outage (missing dates opened); pending and unverifiable dates without notification targets; unreadable notification state; an incident older than 3 days still evaluated; a replaced candidate (the H target stays open); midnight rollover of R1 and E1 conditions recovering on their own episode date; confirmed fake ids; recurrence |
| Eligibility, receipt, mode | an all-rainout day; a resume-only day; a mixed slate with a late cancelled game; a late receipt without a decision; a warning before the entry window (no result) and inside it; a stale heartbeat (unverifiable); both DD legs; wrong-date, wrong-type and wrong-schema records; an intent change yesterday and within today; an untagged active entry cron; an unsupported cron form; non-executing comment text; postseason, no-games and research controls |
| Restart and dependency | churn at each threshold with a fresh heartbeat; first install, reset and gap baselines; a corrupt own history; `--no-churn`, a different `--churn-unit`, `--churn-state`, a missing ping argument, the wrong cadence; a fresh monitor file with empty samples; no heartbeat since start; unparseable `unit` and `ps`; a healthy `uv` parent with its Python child against two independent trees |
| Cron selection (gate 3) | the CLI at 09:55, 23:50, off-grid cutoffs, across midnight and DST, and while the real entry, grading and reconcile writers run (paused at their saves). The actual cron install integration belongs to the whole-watchdog deploy unit and is named there as a prerequisite |
| Confinement (gate 2) | the `fast` job's success, config-error, uncertain-send, deadline, self and ping branches inside the kernel gate. Red controls: a mutant calling the real `proc.run`, the real keychain path, and a same-byte write outside the root |
| Real seams | the process-seam tests in §0.6; DM transport: a missing credential gives `TransportUnavailable` with no subprocess, one attempt per call, body caps, a malformed send response giving no id; ping: never raises, body capped |

**The mutation ledger** is pinned before review. Each mutant names its failure mechanism: removing a validator rule, a class X deadline, a timer anchor, a suppressor, the hard alarm, a completion condition, the eligibility exclusion, a seam allowlist entry, the environment-only credential. Each must make its named test fail on an assertion.

## 9. What is not claimed
- W-postponed reports current status only; never the status at send time.
- W-restart attests liveness, churn, the instance count and the monitor's local configuration and coverage. It does not attest loaded code or poll progress (§7).
- Cross-file results never assert corruption, only overdue completion.
- W-self's dedicated check is operational only after Eric commissions it.
- Nothing here changes a pick, a decision, the streak, the scheduler, the cron or the configuration (R1). Every watchdog write is beneath `data/watchdog/` (R5). There is no authenticated contest request and no cookie read (R6).

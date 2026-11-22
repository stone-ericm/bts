# Review Round 2: D3 and D3b (`24019a9a..428cfa93`)

## Verdict
**SIGN** — Commit `428cfa93` (on top of `90a9b6f0` and `4733a298`) fixes every Round-1 blocking finding from both reviewers on its original inputs, preserves all `K1`–`K20` and appended contract rows, passes all 544 tests in the 7-file focused suite (including `slow`), and introduces no new false-`verified` outcomes.

## Blocking findings
None.

## Notes
- **Per-attempt scope of `legacy.<run>.<attempt>` (`deploy.yml:472,740,1244`):** Keying the Phase-2 legacy-cleanup mark to `(run_id, run_attempt)` matches Amendment 4b R4 and lets later deploys (such as checklist A7's failure drill after B8c's passing deploy) prove stop normally. The trade-off (`M2` in `delta_and_docs.json`) is that if Attempt 1 unloads a legacy scheduler while leaving a residual child whose command line does not match `bts schedule` and an operator launches Attempt 2 without completing `INCIDENT.md §4`'s SSH check (or Attempt 1 passed with such a child under L8/L13), Attempt 2 starts with `bts-scheduler.service` already `not-found` and has no legacy mark of its own.
- **Conservative `unverified` outcomes under the stricter absence rule:** Clean Phase-2 removal of an installed legacy scheduler followed by a later canary failure (`U1`), a one-time failure of the first `ControlGroup` query on a host with no legacy unit (`U2`), and real Debian `systemctl disable`/`stop` exit codes (`1`/`5`) when a `not-found` unit reports an empty cgroup (`U3`) all report `Scheduler stopped: unverified.` even when no scheduler process remains.
- **Skipped mutation run — static test-coverage gaps (`tests/test_deploy_workflow.py`):** Execution of `probe_r2_mutations_v3.py` was skipped per coordinator instruction. Static inspection indicates two required Round-2 edits would likely survive existing assertions: dropping `! legacy_cleaned` in `contain`'s first sample (`r2_req1`, because `prove stop` still enforces `! legacy_cleaned` and `test_d3b_r1_phase_two_legacy_cleanup_marks_the_attempt` discards recovery's call log) and moving Phase 2's `mark` write after `systemctl stop` (`r2_req3`, because `test_d3b_r1_mark_failure_contains_before_the_legacy_stop` checks only that `daemon-reload` did not run, not `disable`/`stop`), alongside Round-1's ledgered `st_uid != 0` check (`r1_own6`) and unobservable `publish contained` read-back match (`r1_own7`).

## 1. Round-1 blocking findings replay
Replayed both reviewers' Round-1 counterexample scripts against `428cfa93` (`probe_r2_replay_round1.py` -> `replay_round1.json`):
- **Codex 1 (URL rewrite):** `/configured/../different-check`, `/a/./b`, `/a/..`, `/%2e%2e/other`, and `/check/%2fother` on deploy and recovery send zero POSTs and record `unverified`; valid POSTs pass `--path-as-is` and `--globoff` (verified via offline `libcurl.4.dylib` URL parser).
- **Codex 2a / Carbon B1 (cgroup on `not-found`):** Codex's `not-found-with-group` and Carbon's `CE1`/`CE1b` (`populated 1` at first or post-stop sample) now inspect the reported cgroup, run legacy stop where applicable, and report `Scheduler stopped: unverified.` across deploy, `contained` recovery, and `begun` recovery.
- **Codex 2b / Carbon B3 (`not-found` with fragment or dangling link):** Both Codex's `legacy-fragment-present` (regular file) and Carbon's `CE3` (`legacy-vanishing` with dangling symlink, empty cgroup) now fail `[[ $legacy == loaded ]]` and report `Scheduler stopped: unverified.`.
- **Carbon B2 (Phase-2 legacy unload before canary failure):** Both `ce2-phase2-unload-populated` and `ce2b-phase2-unload-nogroup` write `/run/bts-deploy/legacy.123.1` (`0600`) and report `Scheduler stopped: unverified.` in deploy and recovery (no `containment re-checked`).
- **Codex 3 / Carbon N2 (two-print report budget):** Replaying Codex's `timing_model.py` `blocked-final-recheck` ends with a single `("report", 277, 279)` entry (`279 s <= 280 s`, no separate `recheck` call); all six `early`/`cap-stress`/`late-successful` window runs finish within `267 s`.

## 2. Must-still-catch table (`K1`–`K20` and appended case)
Reran all 21 independent shim cases in `probe_r2_k_table.py` (`k_table.json`):

| Row | Input run on `428cfa93` | Outcome | Status |
|---|---|---|---|
| K1 | First or every containment clock sample fails (deploy + recovery) | Stops user unit, writes marker, sends POST, records `contained` | PASS |
| K2 | Signals at 790/795/805 s, wait, post-stamp, armed read-126; committing signal | Contains with 0 handler reads; committing uses bounded 15 s read | PASS |
| K3 | User/legacy populated->blank and always-blank loaded cgroups | `Scheduler stopped: unverified.` | PASS |
| K4 | `{other}`, `{one,two}`, `[1-1]`, `[1-2]`, `../other`, `%2e%2e/other`; HTTP controls | Rejected without POST (`unverified`); `--globoff` + `--path-as-is` verified | PASS |
| K5 | Commit stage/rename 17/124/137, read-back 124/137, rename error after commit | Keeps numeric/`timeout` status and contains; exact read-back exits 0 | PASS |
| K6 | Checklist A5 and both B8a holds vs config/cron/enable edits | A5 defers to B8a; both B8a stop+boot-disable holds precede edits | PASS |
| K7 | NUL byte inside zero-exit `contest-state` accepted prefix | `Check contest-state failed (query).`, no `CANARY PASSED` | PASS |
| K8 | Foreign run ID at attempt 1, HTTP 301, real `lockfile`/`mark`, held `flock` | Refuses foreign record, no `-L`, real helpers, waits on lock | PASS |
| K9 | `HUP`/`INT`/`TERM`/`PIPE` before `begun` stage/rename across 3 priors | `DEPLOY NOT STARTED`; prior record, heartbeat, boot link byte-identical | PASS |
| K10 | Signal after `begun` rename; fresh read 126 while `beginning` | Enters `contain interrupted` once (S7, S9) | PASS |
| K11 | Legacy loaded + populated `/legacy-orphan` -> `not-found` (reset, recovery, Phase-2 unload) | Reads `/legacy-orphan` (or sees legacy mark) -> `unverified` | PASS |
| K12 | Empty orphan, nogroup, absent, job, queryfail, dangling link, ghost-group (0/1) | `verified` only when empty/absent; every unproved case `unverified` | PASS |
| K13 | Clock unavailable; slow proof queries hitting 205 s cap | `pgrep` launched (`TERM=11 s >= 5 s`), marker + POST run, stop `unverified` | PASS |
| K14 | Cutoffs in both scripts; valid endpoint read at 248 s and 254 s | `90/180/205/220/240/254/270/278`; curl `TERM` 15 s and 10 s | PASS |
| K15 | Commit rename succeeds, read-back exits 126 | Logs `Check commit-record failed (126).` and contains | PASS |
| K16 | Distinct pre/post cgroup paths (`user` + `legacy`, `(1,0)`, `(0,1)`, `(0,0)`) | Rejects either populated path; checks both paths when empty | PASS |
| K17 | Foreign begin; 4 mismatched `committing` fields; same-ID `beginning` forgeries | Only exact `committing` match exits 0; all others contain | PASS |
| K18 | `contained` recovery with `page=verified` across ok/stop/marker/record faults | No second POST; reproves stop + marker; `re-checked` iff all 3 hold | PASS |
| K19 | Shared prefix and `prove` extraction across both YAML scripts | Byte-identical `prove`; single bounded print; sole `verified` assignments | PASS |
| K20 | `INCIDENT.md` restore order and both B8a `saver-state` invocations | Hold precedes `install restic`/`restic restore`; both commands carry flags | PASS |
| Appended | Commit rename succeeds, read-back 126, `contained` rename fails | Recovery prints `this attempt's record says committed` + `INCIDENT.md §4` | PASS |

## 3. Round-1 fix delta and regression audit
Probed in `probe_r2_delta_and_docs.py` (`delta_and_docs.json`):
- **Legacy mark (`M1`–`M6`):** Tested deploy `(123,1)` -> recovery `(123,1)` -> passing re-run `(123,2)` -> late recovery `(123,1)` (`M1`: attempt 2 passes, late recovery 1 fenced with `cancelled.123.1` and zero service mutations); simulated reboot clearing `/run/bts-deploy` (`M3`: no service mutation); dashboard-only legacy cleanup (`M4`: cleans `bts-dashboard.service` without marking scheduler, `stop=verified`); `SIGTERM` during `mark` (`M5`: `supervise` finishes `mark`, `daemon-reload` skipped, `loaded` proof checks `/legacy-orphan`); and pre-existing symlink/FIFO/`0644`/non-empty vs valid `0600` empty `legacy.123.1` (`M6`: corrupt entries fail closed at `legacy-units` before `disable`/`stop`/`daemon-reload`; valid empty file is idempotent and passes). No order made recovery act on another attempt, left a touched loaded legacy scheduler unmarked, or blocked a passing deploy.
- **Stricter absence rule:** Confirmed conservative `unverified` states `U1`–`U3` (Notes) and searched the reverse direction (`ControlGroup=/` root cgroup lacking `cgroup.events`, dangling symlink at `legacy.123.1`, state/fragment mismatches): zero false-`verified` outcomes found.
- **URL rule:** Standard `hc-ping.com` UUIDs (with/without trailing `/`, quoted), `ping-key/slug` paths, bare hosts, and self-hosted URLs with ports and inner dots (`:8443/v1.2/ping/bts.scheduler~1`) are all accepted and preserved identically by `libcurl`; only paths containing `%` escapes (such as `%20`, `%7e`, `%2e`) or `.`/`..` segments are refused. Inline `fs_code` endpoint checks also passed on Python `3.11.18`.
- **Single final print:** Verified 4 lines normally, 5 lines on all-verified repeat recovery, and clean `exit 1` with `contained` record saved under closed stdout (`EPIPE`).

## 4. New and changed operator sentences
Audited against code and helper scripts (`probe_r2_delta_and_docs.py`, `scripts/install-systemd-hetzner.sh`):

| Location & changed sentence | Worst state if followed literally |
|---|---|
| `INCIDENT.md:120-121` — *"On the (re)built box, first confirm the scheduler is stopped and its local boot link disabled, and keep it so through restore and activation checks. Then install restic..."* | Stops any running scheduler and disables its boot link before `restic restore` or `sync-from-r2` touches `data/picks` or `data/health_state`; prevents mid-restore picks or state clobber. |
| `INCIDENT.md:131-133` — *"`bash scripts/install-systemd-hetzner.sh` ... copies the user unit templates and reloads the user manager. Follow §4 and activate the scheduler through §5's checked deploy."* | Verified `install-systemd-hetzner.sh` only installs unit files and runs `daemon-reload` (never `enable` or `restart`); activation stays gated behind §5's canary deploy. |
| `INCIDENT.md:205` — *"During initial containment, and on recovery whenever the attempt record does not hold a verified 2xx (a real earlier 2xx is sent again if its record rewrite failed)."* | Duplicate `/fail` alert if prior `contained` rewrite failed; if the record is absent/foreign/unreadable/`committed`, recovery sends no POST and prints its exit line, which §4 routes to Eric. |
| `CLAUDE.md:135` (`L10`) — *"The box can commit after Actions fails or cancels. Recovery then reports that this attempt's record says committed."* | If a `committed` record survived a failed `contained` rewrite after read-back error, treating it as a record statement (with §4's log + SSH check) prevents assuming the scheduler is running. |
| `docs/ops/2027-season-start.md:330` (`B8c`) — *"...keep both services stopped during maintenance, with the scheduler's local boot link disabled through the switch from maintenance to normal boot."* | Prevents `loginctl` linger from auto-starting the October `619abda7` user scheduler when rebooting from rescue/maintenance into normal boot. |

## 5. Mutations
All 28 scratch mutation targets in `probe_r2_mutations_v3.py` (the 17 Round-1 mutants adapted to `428cfa93`, the 6 required Round-2 mutants, and 5 new reviewer mutants) passed `--precheck-only` target validation. Per coordinator instruction after `GO`, the wrapped pytest mutation run was skipped. By inspection of `tests/test_deploy_workflow.py`, 4 of the 6 required Round-2 mutants (`r2_req2` drop mark in proof, `r2_req4` drop `--path-as-is`, `r2_req5` accept `%`, `r2_req6` drop `[[ $legacy == loaded ]]`) directly falsify `test_d3b_r1_*` assertions, while `r2_req1` and `r2_req3` lack direct assertions (see Notes).

## What I ran / What I did not run
- **Ran:** `probe_r2_replay_round1.py` (`replay_round1.json`), `probe_r2_k_table.py` (`k_table.json`), `probe_r2_delta_and_docs.py` (`delta_and_docs.json`, including `libcurl.4.dylib` URL checks and Python `3.11.18` inline `fs_code`), `probe_r2_mutations_v3.py --precheck-only`, and the wrapped 7-file focused pytest suite (`pytest_focused.log`: `tests/test_deploy_workflow.py`, `test_deploy_canary.py`, `test_heartbeat_staleness.py`, `test_heartbeat_churn.py`, `test_lightgbm_optional.py`, `test_web_saver.py`, `test_wr_b_contract.py` with `-m "slow or not slow"` — **544 passed in 1609.23s**).
- **Did not run:** Pytest execution of `probe_r2_mutations_v3.py` (skipped per coordinator instruction), the full fast pytest suite, real host/SSH/systemd/cgroupfs/reboot execution, or live network/provider paging.

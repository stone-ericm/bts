# Kickoff: whole-range change set D, the deploy workflow's failure path (W12, W13) (draft 1, for the contract check)

Written 2026-11-19 by `bts-wr-fresh2` on Eric's approval (FINDINGS.md, "Eric, 2026-11-19 evening": "yes" to a failed live deploy stopping outward sending and paging him, with no automatic rollback, and to a failed sync or first restart not skipping the recovery branch). Tier A (deploy and scheduler start/stop behaviour). Tested ONLY by running the workflow's own shell under command shims (the method of `slice-2-codex/deploy_probe.py`, `workflow()` and `shim()`): nothing runs against the real box (it is deleted), nothing is pushed to `deploy`, no GitHub workflow is dispatched.

You work in `~/projects/bts-work/wr-fix-deploy`, branch `fix/wr-deploy`, cut from `origin/main` at the commit named when you are started. Tests first (RED on the base), then the change, then the focused tests, then the fast suite on the COMMITTED tree. One commit. Same run rules as the other whole-range change sets (`kickoff-fix-c2.md` first paragraph: `yield_run.sh`, one wrapper run at a time, patch tool for files, no pattern kill, bounded reads, commit before the fast suite, failing set exactly `~/bts-runs/gate-failed.txt`). **If an existing test asserts behaviour this contract changes, STOP and ask, naming the test.**

**Read first:** `.github/workflows/deploy.yml` (the whole `script:`); `slice-2-codex/report.md` blocking findings 2 and 3 and "Decide what the deploy canary should certify", and its `deploy_probe.py`; `scripts/systemd/bts-scheduler.service`; `scripts/check_heartbeat.py`; `scripts/cron-setup-hetzner.sh` (the cron table); `INCIDENT.md` §4 and §5; `CLAUDE.md` "Deployment"; `docs/ops/2027-season-start.md` row A7.

## What a failed deploy does today (base `.github/workflows/deploy.yml`, step by step)
1. `set -e`; `cd`; `OLD_SHA=$(git rev-parse HEAD)`; `git config … safe.directory`; `git fetch origin`; `git reset --hard "$TARGET_SHA"`.
2. `uv sync --locked --extra model` (as `bts`).
3. Self-heal of legacy system units (only if `/etc/systemd/system/bts-scheduler.service` exists).
4. `systemctl --user restart bts-scheduler bts-dashboard`; `NEW_SHA`.
5. `sleep 30`; `set +e`; canary: scheduler active (or a clean exit waiting for `Restart=always`), dashboard active, dashboard HTTP 2xx on the tailnet address.
6. Canary failed: `git reset --hard "$OLD_SHA"`, `uv sync`, restart both units, `sleep 15`, stamp `data/.last_deploy_iso` if both active, else print `CRITICAL: rollback also unhealthy`; `exit 1`.
7. Canary passed: stamp `data/.last_deploy_iso`; exit 0.

**Defects.** (W13) Under `set -e`, a non-zero exit at step 1's reset, step 2, step 3 or step 4 ends the script before step 5: the candidate (or a half-synced tree) stays checked out, the scheduler is in whatever state the failed command left (still running the OLD code in memory after a failed sync; down or crash-looping after a failed restart), nothing is rolled back and nobody is paged except by GitHub's own failed-run notice. (W12) When the canary fails, step 6 restores `OLD_SHA`. On a 2027 box restored from snapshot `446156433`, `OLD_SHA` is `619abda7` (or another pre-2027 commit), which cannot serve the season (2026 day count) and damages state the new code then refuses (a fetch rewrites the automatic file as v1; scoring drops `season` and `applied_dates` from `streak.json`; `slice-2-codex/` rollback replays). Even a good earlier 2027 commit is not known to read state the candidate wrote.

## Design (the lead's proposal; the contract check rules on it)
| # | Required |
|---|---|
| D1 | **No automatic rollback, in any mode.** The workflow never resets the box to `OLD_SHA` or any other commit than `TARGET_SHA`. `OLD_SHA` is still captured and printed, for the operator. Same behaviour whether the box is live (`pick_delivery` dm or public, `entry_intent = "enter"`) or in research mode: the script reads no configuration to decide (a mode switch is one more thing that can be wrong; research mode loses only its research capture while stopped). |
| D2 | **Two phases.** Before the first command that changes the box (the `git reset --hard "$TARGET_SHA"`), a failure exits non-zero with the box unchanged and the running scheduler untouched (it is the code that was already running), printing `DEPLOY NOT STARTED — <step> failed; the box is unchanged (still <OLD_SHA>)`. From the reset onward, ANY non-zero exit of any command (the reset itself, the sync, the self-heal block, the first restart, `NEW_SHA`, and each canary check) goes to ONE failure function; nothing in that phase can exit the script except through it. |
| D3 | **The failure function** (all of it best-effort, each step's own failure recorded and printed, none able to skip a later step): (a) print `DEPLOY FAILED at <step> — target <TARGET_SHA short> is checked out; NO automatic rollback; the scheduler is being stopped` and the reasons; (b) `systemctl --user stop bts-scheduler` (stop only: not disable, not mask; the dashboard is left as it is); (c) mark the heartbeat: write `data/.heartbeat` as user `bts`, atomically (temporary file in `data/` then rename), with `{"state": "deploy_failed", "timestamp": <UTC ISO now>, "target_sha": <short>, "old_sha": <short>, "step": <step>}`; (d) page now: one `curl -fsS -m 10` POST to `$BTS_SCHEDULER_HEARTBEAT_PING_URL/fail` read from `/home/bts/projects/bts/.env` (as user `bts`; the URL and the `.env` content never printed), body `deploy failed at <step>, target <short>, scheduler stopped`; (e) verify: print whether the scheduler reads inactive, whether the heartbeat says `deploy_failed`, whether the ping returned 2xx; (f) do NOT stamp `data/.last_deploy_iso`; (g) `exit 1`. |
| D4 | **`scripts/check_heartbeat.py`:** a heartbeat with `state == "deploy_failed"` is stale, with the reason `deploy failed at <step> (target <sha>): scheduler stopped by the deploy workflow; see INCIDENT.md`. So the 5-minute cron pings `/fail` every run until a scheduler start rewrites the heartbeat (healthchecks stays down, not a single flap). Every other state is judged exactly as today. |
| D5 | **The success path is unchanged** byte for byte in the commands it runs and what it stamps (the shim call log of a passing canary is identical to the base's). A successful later deploy `restart`s a stopped scheduler (systemd starts a stopped unit on `restart`), whose start rewrites the heartbeat. |
| D6 | **Docs.** `INCIDENT.md` §4 (a failed deploy now leaves the target checked out and the scheduler STOPPED; how to recover: fix forward and push to `deploy`, or on the box `systemctl --user start bts-scheduler` once the cause is understood; never reset the box to the pre-deploy SHA unless that SHA is known to serve the current season with the state on disk) and §5 (`git push origin main:deploy` line, and the "Blunt rollback" block gains the same warning: a pre-2027 SHA such as `619abda7` cannot serve 2027); `CLAUDE.md` "Deployment" (the "Canary + auto-rollback" bullet); checklist row A7. Ledger W17 (the canary still passes on a box that starts, pages and idles) as known and not fixed. |

## Must still happen (each a test under shims; RED on base where marked)
| Case | Required outcome |
|---|---|
| Canary passes (both units active, HTTP 200) | exit 0; call log identical to base; `.last_deploy_iso` stamped; no stop, no heartbeat write, no ping |
| Scheduler completed cleanly and waits for `Restart=always` (activating / auto-restart / success / 0) | passes as today |
| `git fetch` fails | exit 1, `DEPLOY NOT STARTED`; no reset, no restart, no stop, no ping; scheduler untouched (RED on base? base also exits 1 untouched: GREEN control) |
| `git reset --hard "$TARGET_SHA"` fails | failure function: stop, heartbeat, ping, exit 1; no reset to `OLD_SHA` (RED on base: base exits with nothing) |
| `uv sync` fails (W13) | failure function (RED on base) |
| Self-heal block fails (legacy unit present, `daemon-reload` fails) | failure function (RED on base) |
| First restart fails (for example the 6a live gate's `refusing to start`, exit 2) (W13) | failure function (RED on base) |
| Scheduler crash-looping at the canary (activating / auto-restart / exit-code) | failure function; no `OLD_SHA` reset (RED on base: base rolls back) |
| Dashboard inactive / HTTP 503 / curl timeout, scheduler healthy | failure function: the scheduler IS stopped (the deploy is not certified) (RED on base: base rolls back) |
| Inside the failure function: `systemctl stop` fails; the heartbeat write fails (`data/.heartbeat` is a directory; `data/` not writable); `.env` missing or has no ping URL; curl fails or times out | every later step still runs; each failure printed; exit 1 |
| Research-mode box (`pick_delivery = "private"`, `entry_intent = "research"`) | identical behaviour to a live box in every case above |
| `check_heartbeat.py` on a `deploy_failed` heartbeat | stale, `/fail` pinged, exit 1, every run; on any other state byte-identical output to base |
| After a failed deploy, a later successful deploy | scheduler restarted; heartbeat no longer `deploy_failed` once it starts (show with a shim or name the scheduler code that writes the heartbeat at start) |

## Standing hostile-input rows (say how each is tested or why it cannot apply)
- **An interrupt mid-step:** the SSH session drops, or the run is cancelled, inside phase 2 (the shell receives SIGHUP/SIGTERM): what is left (candidate checked out, scheduler in any state) and who is paged (only GitHub's failed-run notice; the heartbeat cron if the scheduler then stalls). Decide whether a `trap` on HUP/TERM should run the failure function, and test it if so.
- **A malformed or non-regular file:** `data/.heartbeat` is a directory, a symlink or a FIFO when the failure function writes it (the rename must not follow a symlink out of `data/`, and must not block on a FIFO); `.env` unreadable.
- **Midnight ET and the year boundary:** the script has no date logic except the `.last_deploy_iso` stamp and the heartbeat timestamp (both UTC): say so.
- **A failed or unconfirmed send:** the page (curl) fails: the heartbeat state still pages within five minutes; if the venv is broken so `check_heartbeat.py` cannot run either, the healthchecks.io check's own missed-ping timeout is the last signal (it is configured on healthchecks.io, not in the repo: the contract check says whether that is acceptable and what the checklist must tell Eric to verify).
- **A restart mid-step:** the box reboots during phase 2 or after a failed deploy: the enabled scheduler starts on whatever tree is checked out (the candidate, possibly half-synced). Decide whether D3's "stop only" is enough or the unit must also be prevented from starting at boot (and then how the next successful deploy re-enables it).

## Questions the contract check must answer
1. Is "stop the scheduler, leave the candidate checked out, leave cron as installed" the right meaning of "stop outward sending"? List every cron line in `scripts/cron-setup-hetzner.sh` (both intents) and every systemd timer in `scripts/systemd/`: which can send anything outward (DM, public post) or write contest or streak state while the scheduler is stopped on a failed candidate, and whether any must be stopped too.
2. Is the heartbeat-state page (D3c, D4) sound: does anything else write `data/.heartbeat`; does the scheduler rewrite it on every start before it can sleep for hours; does `check_heartbeat.py`'s churn check or any other reader misbehave on the new state?
3. Is D2's phase boundary right (is a failed `git fetch` or `rev-parse` really "box unchanged")?
4. What exactly does the operator read, and is every sentence safe if followed in the worst state (the lesson of `STOP-C2.md` and `STOP-C3V.md`): the workflow log lines, the ping body, the heartbeat reason, and the INCIDENT.md text.
5. Can the test harness run the script under `/bin/bash` on macOS (bash 3.2) and on the CI runner's bash alike, and does extracting it from the YAML stay exact?

## Tests
`tests/test_deploy_workflow.py` (new): extracts the `script:` block from `.github/workflows/deploy.yml` exactly (fail if the extraction does not find exactly one block), runs it with `/bin/bash` and command shims (`sudo`, `git` via `sudo`, `uv`, `systemctl`, `curl`, `sleep`, `id`, `tailscale`) that log every call, under a temporary root, with a timeout on every run; one test per row of both tables. A test for D4 in the existing heartbeat test file or a new one. Self-contained; no network; nothing outside the temporary root written.

## Handover
`HANDOVER-WR-DEPLOY.md` in the worktree root (not committed): the commit; the before and after of the script; per row the test and whether it was RED on the base; the hostile-input rows; the docs changed; what you ran with counts. Then print `WRD-<first 8 hex characters of the sha256 of HANDOVER-WR-DEPLOY.md>` on its own line and `DONE`.

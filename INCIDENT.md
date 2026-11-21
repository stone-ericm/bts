# INCIDENT.md — Remote / Phone Incident Runbook

**Read this if BTS looks broken and you're fixing it remotely** — e.g. a Claude Code
*cloud* session opened from the Claude mobile app, or Eric driving from away-from-keyboard.
It tells you what you can fix from where you are, what you can't, and the exact commands.

## 0. First: where are you running? This decides everything.

- **Cloud session** (claude.ai/code or the Claude mobile app): you're in an
  Anthropic-managed sandbox. You have **this GitHub repo and nothing else** — no SSH to the
  production box, no Tailscale, no access to Eric's Mac, Keychain, or global memory. You can
  fix code and ship it through GitHub. This is the usual case for "Eric only has his phone."
- **Local / Remote Control** (running on Eric's Mac): you have his SSH key, Tailscale, and
  full access to bts-hetzner. You can do everything below *plus* the on-box items in §3.

Quick self-check: if `git remote -v` works but you cannot reach `http://bts-hetzner:3003`
or `ssh root@bts-hetzner`, you're in a **cloud sandbox** — stay inside the §2 "CAN do" list.

## 1. How BTS deploys (so you don't break it)

- Production runs on **bts-hetzner** (Hetzner VPS): `bts-scheduler` + `bts-dashboard`
  systemd `--user` units, run as user `bts`.
- **Deploy = push to the `deploy` branch.** `.github/workflows/deploy.yml` runs on a
  GitHub-hosted runner and SSHes into the box using a key stored in **GitHub Actions secrets**
  (`HETZNER_SSH_KEY`) — *not* on anyone's laptop. **That's why you can deploy from a cloud
  sandbox with zero SSH access of your own.**
- The workflow checks the target and service manager, disables the scheduler's local boot link
  as a soft hold, resets to the run's tested commit, syncs, restarts both user services, waits
  30 seconds and runs the canary. It checks both units, dashboard HTTP and the contest-state
  read for a live configuration. After authorization the remote script attempts containment on
  a detected failure. Recovery is eligible after a failed or cancelled deploy step. Verify the
  result in Actions and follow INCIDENT.md §4 if containment cannot be confirmed.
  `CANARY PASSED` records the checks at that moment. Verify pick delivery and MLB entry separately.
- **Do NOT** `systemctl restart` on the box after pushing — the workflow already does it.
- `origin/main` and `origin/deploy` are **independent**. Deploy ships *whatever ref you push
  to `deploy`*. Normal path is `git push origin main:deploy`; you can also push a branch with
  `git push origin HEAD:deploy` (see the ⚠️ footgun in §5).

## 2. What you CAN do from a cloud session

- ✅ Diagnose a **code bug** against the repo + tests.
- ✅ Write a fix on a branch.
- ✅ Ship an approved fix through `deploy` (§5).
- ✅ Revert (§5).
- ✅ **Re-run a deploy** with no code change: GitHub → Actions → "Deploy to Hetzner" →
  **Run workflow** (`workflow_dispatch`) — **select the `deploy` branch in the dropdown**
  (defaults to `main`, which fails at the environment gate: the root SSH key lives in the
  `production` environment whose branch policy allows only `deploy`; rotated 2026-07-10).
  Select `deploy`. A new dispatch uses that ref's current SHA and runs the whole deploy.
- ✅ Read the Actions result and the deploy and recovery lines. A failure can occur before the canary.

Run tests before shipping:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
```

Model/predict tests need `--extra model` → LightGBM + libomp, which may not install in the
sandbox. Tests that fail to collect for that reason are fine to ignore — the **contest / cli /
health** suites run without the model and validate most fixes. Don't block a fix on the model
tests.

## 3. What you CANNOT do from a cloud session — ESCALATE, don't flail

If the problem is any of these, **stop. Tell Eric precisely what to run** on a machine with
SSH + Tailscale. Don't burn turns trying to reach the box; you can't.

| Can't do from cloud | Why |
|---|---|
| SSH into bts-hetzner; read `journalctl` / systemd state | no key, no egress to the box |
| View the dashboard `http://bts-hetzner:3003` | Tailscale-only |
| Edit on-box config: `~/.bts-orchestrator.toml`, `.env` | live on the box, not in the repo |
| **Re-capture expired leaderboard cookies** (`scripts/capture_bts_cookies.py`) | interactive + on-box — the known human-in-the-loop dependency |
| Inspect/repair on-box data under `/home/bts/...` | on the box |

**The cookie case is the big one.** If picks have frozen and the signal points at leaderboard
auth / login / cookies, that is **not a code bug** — it's expired
`~/.bts-leaderboard-cookies.json` on the box, and there is no cloud fix. Escalate with a
ready-to-run instruction, e.g.:

> "BTS picks are frozen on a leaderboard-auth/cookie failure — this needs the box, which I
> can't reach from the cloud sandbox. On a machine with SSH to bts-hetzner, as the `bts` user:
> `python scripts/capture_bts_cookies.py` to re-capture cookies."

## 4. Playbook by symptom

**Triage rule:** *Can you reproduce or locate it in the repo + tests?* → fix forward (§2).
*Is it about the box's runtime, auth/cookies, config files, or network?* → you can't reach it
from the cloud (§3) → escalate.

- **A deploy failed.** Read the Actions result and the deploy and recovery logs.
  `DEPLOY NOT STARTED` describes that invocation. Read recovery separately and verify the stop,
  heartbeat and HTTP results. The box may have committed even if Actions lost the log.
  If any result is unverified or absent, the run was cancelled, or the reports disagree, contact
  Eric before retrying. He checks both scheduler scopes and the boot link by SSH. The old or
  candidate scheduler can still be running.
  Preserve pick, receipt, saver and streak files. Verify today's ET slate, the original Bluesky
  delivery and the MLB entry read-only. Leave the dashboard's saver controls and MLB entries
  untouched, do not enter a pick from a preview, and never resend. If delivery or entry remains
  uncertain, keep the scheduler stopped once confirmed and have Eric resolve the uncertainty
  before attempting recovery. Apply the same checks and restart window to a dashboard fault.
  During deploy recovery the dashboard is for inspection.
- **Picks frozen / none posted:** check the health DM Eric received and the Actions log. A code
  exception in a recent change → fix forward. Auth / cookie / box → **escalate** (§3).
- **Bad pick logic, service otherwise up:** code fix → redeploy, but **respect the leakage
  safety rules in `CLAUDE.md`** — never ship feature changes without the `shift(1)` guards +
  `scripts/leakage_audit.py`. A wrong-but-running model is not an emergency; don't make it
  worse with a rushed change.
- **Dashboard down, picks fine:** a redeploy (or `workflow_dispatch`) restarts `bts-dashboard`.
  If that doesn't fix it, it's on-box → escalate.
- **Box/filesystem loss — operational-state restore (audit F5, needs SSH):** encrypted restic
  backups of `data/picks` + `data/health_state` (ops set, 3h cadence) and
  leaderboard/hetzner_results/external (archive set, daily) live in the R2 bucket under
  `restic/`. On the (re)built box: install restic (`bash scripts/install-restic-hetzner.sh`),
  put `RESTIC_PASSWORD` into `.env` (copy lives in Eric's Mac Keychain:
  `security find-generic-password -a claude-cli -s r2-bts-restic-password -w`) alongside the
  `R2_*` creds, then `bts backup restore-drill --target /tmp/restore-drill` to prove the
  snapshot reads back (verifies streak + saver flag + contest ledger + decision provenance),
  and restore for real **by the snapshot ID the drill printed — never `latest`** (`latest
  --tag ops` can select a partial rc=3 snapshot from a failed run; the drill prefers
  `last_success_snapshot_id` from the backed-up status file):
  `~/.local/bin/restic restore <SNAPSHOT_ID> --target /` — restic re-creates
  the original absolute paths (`/home/bts/projects/bts/data/...`). Parquets/models come from
  the separate artifact sync: `bts data sync-from-r2`. Systemd units: repo templates via
  `bash scripts/install-systemd-hetzner.sh` (audit F12), then enable + start deliberately.

## 5. Deploy and revert commands

**Normal ship** (keeps `main` as the source of truth that gets deployed):

```bash
git checkout -b fix/<short-desc>
# make the fix + add/adjust tests
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
git commit -am "fix: <what and why>"
git push origin fix/<short-desc>     # open a PR so Eric can eyeball it from his phone
# after it's merged to main:
git push origin main:deploy
```

This runs the gate, deploy and canary. Follow §4 for failed or cancelled runs.
The workflow leaves the reached checkout in place.

**Emergency lane** (picks actively broken, fix is small + obvious, Eric ok'd it in chat) —
ship the branch straight to `deploy`:

```bash
git push origin HEAD:deploy
```

> ⚠️ **Then land the same fix on `main`** (merge the PR). `deploy` and `main` are independent,
> so the next normal `main:deploy` will **revert** anything that's on `deploy` but not on
> `main`. An emergency branch-ship is not done until main has it too.
> **Use a true merge, not a squash**: a squash-merge means the emergency commit is not an
> ancestor of `main`, so the reconciling `main:deploy` push is non-fast-forward and the
> `protect-main-deploy` ruleset will reject it.

**Revert a bad change** (preferred — history keeps moving forward):

```bash
git checkout main && git pull
git revert <bad-sha>
git push origin main            # keep main canonical
git push origin main:deploy     # ship the revert
```

Deploy an older commit only with Eric's explicit approval and a rehearsal against today's
on-disk state, dependencies and installed jobs. Do not deploy `619abda7` or other code without
2027 support to a 2027 box. Prefer a revert commit through the normal ship.

For a code failure, fix forward with a tested commit. For a contest-state refusal, use the
scheduler's configured picks directory, confirm today's ET date and MLB values, and correct
the state under checklist B8a. Run `bts season-contract status --picks-dir <scheduler picks dir>`
and read `live decision now: accepted`. If approved for today's remaining verified lineup and
submission window, rerun the same tested SHA. Use Re-run jobs on the original run if eligible,
or dispatch `deploy` after confirming it still points to that SHA. Check the current ET state
again if the day changed. Never start or restart the scheduler by hand to retry a failed
deploy. If no usable window remains, tell Eric, check that the scheduler is stopped, and
record the missed day.

After a verified recovery, check missed grading and observations read-only, and backfill only
in verified chronological order with Eric. A new dispatch of a moved branch can ship a
different commit.

## 6. Where signal comes from when you're remote

- **Bluesky DM health alerts** — CRITICAL alerts from `bts.health.runner` DM Eric; he'll relay
  the text, which usually names the failing subsystem.
- **GitHub Actions log** of the last deploy — canary failure reasons, in plain text.
- **The repo + tests** — reproduce code bugs here.
- You **cannot** see the live dashboard or box logs from the cloud. Don't guess at on-box
  state — ask Eric or escalate per §3.

| Channel | When |
|---|---|
| GitHub failed-run notice | Depends on GitHub recording a failure and Eric's settings. A preflight failure, forced cancellation or failed SSH can leave the old process running with no page. |
| Immediate scheduler-heartbeat `/fail` POST | During initial containment, and on recovery if the prior response was unverified. A 2xx confirms the POST response. Check the provider's history and Eric's receipt separately. |
| Heartbeat cron | Every five minutes while the failure marker remains or the heartbeat is stale, once the stock-Python line is installed. |
| Healthchecks missed pings | The liveness check when the box, cron or egress is down. The heartbeat check when its checker cannot run. Both depend on provider schedule, grace and integration. |

Cron and timers continue.

- Scoring starts at 01:00 and may run until 06:00 ET. Reconcile runs at 02:00 and 07:40.
  Contest fetch runs at 01:10, 02:10, 10:30 and 13:30. Each can write against the reached tree.
- The 03:00 data build can run preview and sync assets. The 07:45 park-drag refresh,
  23:30 shadow update, five-minute lineup collector, 30-minute static capture, and analysis
  timers continue.
- Ops backups run at :20 every three hours. Archive runs at 04:50 and pruning at 05:35 Sunday.
- Entry checks every 15 minutes from 10:00 through 23:45, the 08:30 day-outcome job, and the
  contest fetch can DM according to the installed cron and intent. The October fetch can DM
  in research. These jobs send no pick.

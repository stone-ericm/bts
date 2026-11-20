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
- The on-box owner arms a durable hold before resetting to the triggering SHA. It stops
  the scheduler, drains writers, syncs locked dependencies and checks candidate readiness
  and the dashboard. In enter mode the canary requires usable contest state for the sampled
  ET date. The owner starts production and durably clears the hold after verification.
  Failure keeps the candidate checkout in place and closes the guarded jobs. Recovery is
  by verified fix forward.
- **Do NOT** `systemctl restart` on the box after pushing — the workflow already does it.
- `origin/main` and `origin/deploy` are **independent**. Deploy ships *whatever ref you push
  to `deploy`*. Normal path is `git push origin main:deploy`; you can also push a branch with
  `git push origin HEAD:deploy` (see the ⚠️ footgun in §5).

## 2. What you CAN do from a cloud session

- ✅ Diagnose a **code bug** against the repo + tests.
- ✅ Write a fix on a branch.
- ✅ **Ship it** by pushing to `deploy` after the checks in §5.
- ✅ **Re-run a deploy** with no code change: GitHub → Actions → "Deploy to Hetzner" →
  **Run workflow** (`workflow_dispatch`) — **select the `deploy` branch in the dropdown**
  (defaults to `main`, which fails at the environment gate: the root SSH key lives in the
  `production` environment whose branch policy allows only `deploy`; rotated 2026-07-10).
  This is a full deploy of the selected `deploy` SHA, subject to §5.
- ✅ **Read the last deploy's Actions log** — the canary prints exactly which check failed.
  This is your best remote diagnostic.

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

- **Deploy failed or its result is unknown.** Read the Actions result, then use the run ID
  to inspect the on-box result and `bts-deploy-hold status` read-only. Status exit 1
  is expected for a hold. If the result or containment is uncertain, contact Eric before
  any retry. Keep the hold and preserve pick, receipt, saver and streak files. Verify
  today's ET slate, the original Bluesky delivery and MLB entry read-only. Use the dashboard
  for inspection. A missing receipt, preview or alert authorizes no entry or resend. A
  delivery may already have been admitted during activation.
- **Contest-state refusal.** Read `bts season-contract status`. On a dated day, verify MLB
  has settled the prior slate before entering today's observation. Rerun only inside a
  verified window. If delivery or entry is uncertain, or the window is too short, keep
  the hold and report to Eric.
- **Partial sync.** Repair dependencies through a deploy of the intended revision. Check
  stop and paging evidence. A 2xx response records HTTP acceptance. If checkout monitoring
  is unavailable, consult the rehearsed missed-ping path.
- **Hand release (Eric only).** Confirm the intended revision, current ET state, prior
  delivery and MLB entry and the recovery window before running
  `bts-deploy-hold release --by-hand --reason "<reason>"`. The command checks the clean
  checkout, loaded guard, installed cron, canary and offline locked uv check. Keep the hold
  if any prerequisite is uncertain. Cron and boot can run as soon as the hold clears.
  Recheck the window before using `systemctl --user start bts-scheduler` as `bts`.
- **Calibration warning after verified activation.** Record the warning and inspect
  `data/.last_deploy_iso` read-only. The previous cutoff can remain in use.
- **After verified recovery.** Review grading, reconciliation and observation gaps read-only.
  Backfill in verified chronological order after Eric checks the dates, the applied results
  and the current streak.
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

## 5. Deploy and recovery

**Normal ship** (keeps `main` as the source of truth that gets deployed):

```bash
git checkout -b fix/<short-desc>
# make the fix + add/adjust tests
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
git commit -am "fix: <what and why>"
git push origin fix/<short-desc>     # open a PR so Eric can eyeball it from his phone
# after it's merged to main:
git push origin main:deploy          # ship inside the verified window
```

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

Eric verifies the intended revision, season state, configured delivery, prior send and
MLB entry and a safe sleep window before shipping or retrying. A revert must serve the
current season with today's state and installed jobs. Commit `619abda7` cannot serve 2027.

For writer contention, wait for completion and verify a new window. For a refused window,
recheck the current slate and prior delivery and entry before retrying. A rerun through
`workflow_dispatch` performs a full deploy. When the heartbeat is absent, stale or unreadable,
Eric supplies `recover_sha`, the exact 40-hex target, after verifying prior delivery and entry
and a safe window. That input authorizes only the specified SHA.

## 6. Where signal comes from when you're remote

- **Bluesky DM health alerts** — CRITICAL alerts from `bts.health.runner` DM Eric; he'll relay
  the text, which usually names the failing subsystem.
- **GitHub Actions log** of the last deploy — canary failure reasons, in plain text.
- **Scheduler heartbeat.** A hold, a missing fresh heartbeat after release or missed pings
  can take this check down. Box liveness remains independent. The 08:30 day-outcome DM
  reports yesterday by its own evidence.
- **The repo + tests** — reproduce code bugs here.
- You **cannot** see the live dashboard or box logs from the cloud. Don't guess at on-box
  state — ask Eric or escalate per §3.

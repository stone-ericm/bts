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
- The workflow hard-resets the box to the pushed commit, runs `uv sync --extra model`, restarts
  both services, waits 30s, then runs a **canary**: scheduler healthy + dashboard active +
  dashboard HTTP 200. **If the canary fails it resets the box to the commit that was checked
  out before the deploy and restarts both services again.** That is NOT a guarantee that the
  box is healthy afterwards (whole-range review W12, W13, W17; the workflow change is deferred
  to January 2027):
  - the earlier commit must itself be able to serve the current season. The October 2026 code
    (`619abda7`, what a restored offseason snapshot boots with) cannot serve 2027 and damages
    state the current code then refuses. The first deploy onto a restored box follows
    `docs/ops/2027-season-start.md` row B8c, nothing else;
  - if the sync or the first restart fails, the job stops BEFORE the canary and nothing is
    rolled back: the new code is checked out and the services are in whatever state the
    failure left them;
  - the canary passes when the scheduler merely runs. It does not show that a pick can be made
    (a scheduler with no usable contest state starts, pages and idles).
  After any failed deploy, find out which commit the box has and whether both services run
  (§3: that needs SSH) before doing anything else.
- **Do NOT** `systemctl restart` on the box after pushing — the workflow already does it.
- `origin/main` and `origin/deploy` are **independent**. Deploy ships *whatever ref you push
  to `deploy`*. Normal path is `git push origin main:deploy`; you can also push a branch with
  `git push origin HEAD:deploy` (see the ⚠️ footgun in §5).

## 2. What you CAN do from a cloud session

- ✅ Diagnose a **code bug** against the repo + tests.
- ✅ Write a fix on a branch.
- ✅ **Ship it** by pushing to `deploy`. The canary and the automatic rollback catch a service
  that will not start; they do not make a bad deploy safe (§1). Every deploy restarts the
  scheduler: in season, do not deploy while a pick is being chosen or sent, or during grading.
- ✅ **Roll back / revert** (§5).
- ✅ **Re-run a deploy** with no code change: GitHub → Actions → "Deploy to Hetzner" →
  **Run workflow** (`workflow_dispatch`) — **select the `deploy` branch in the dropdown**
  (defaults to `main`, which fails at the environment gate: the root SSH key lives in the
  `production` environment whose branch policy allows only `deploy`; rotated 2026-07-10).
  Restarts services off the current `deploy` ref.
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
auth / login / cookies, that is **not a code bug**, and there is no cloud fix. What to do
depends on what the failure message itself says, because the contest account is Eric's real
one and extra login traffic can get it blocked:

- **"RATE-LIMITED" (a 403 or 429 anywhere):** back off. Do NOT re-run the command, do NOT
  re-capture cookies. Tell Eric; it is his decision when anything runs again.
- **"transient" / "outage-shaped" / "cookie refresh NOT indicated":** wait for the next
  scheduled run. Re-capture only if the same failure persists across runs.
- **"auth/cookie error — refresh via capture_bts_cookies.py":** the login has expired
  (`~/.bts-leaderboard-cookies.json`). Only then escalate with a ready-to-run instruction, e.g.:

> "BTS reports an expired contest login (auth/cookie error) — this needs Eric: the cookies are
> re-captured interactively on his Mac with `python scripts/capture_bts_cookies.py` and copied
> to the box. I can't do that from the cloud sandbox."

## 4. Playbook by symptom

**Triage rule:** *Can you reproduce or locate it in the repo + tests?* → fix forward (§2).
*Is it about the box's runtime, auth/cookies, config files, or network?* → you can't reach it
from the cloud (§3) → escalate.

- **A deploy failed / canary rolled back:** open Actions → last "Deploy to Hetzner" run; the
  log names the failed check, or shows that the job stopped before the canary (then nothing
  was rolled back, §1). Either way **tell Eric which commit the box is now on** (the log's
  `Pre-deploy SHA` line is where a rollback lands) before shipping anything else. Code error
  in a recent commit → fix forward. If it says *"rollback also unhealthy — manual intervention
  required"*, or the pre-deploy commit is from before the current season, the box needs a
  person → **escalate** (needs SSH).
- **Picks frozen / none posted:** check the health DM Eric received and the Actions log. A code
  exception in a recent change → fix forward. Auth / cookie / box → **escalate** (§3).
- **Bad pick logic, service otherwise up:** code fix → redeploy, but **respect the leakage
  safety rules in `CLAUDE.md`** — never ship feature changes without the `shift(1)` guards +
  `scripts/leakage_audit.py`. A wrong-but-running model is not an emergency; don't make it
  worse with a rushed change.
- **Dashboard down, picks fine:** leave it until the scheduler is asleep. A redeploy (or
  `workflow_dispatch`) restarts `bts-dashboard` but also restarts the scheduler, and a failed
  canary then resets the code (§1): never use it as a dashboard restart while a pick is being
  chosen or sent, or during grading. On-box, `systemctl --user restart bts-dashboard` touches
  only the dashboard → escalate for that.
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
  `bash scripts/install-systemd-hetzner.sh` (audit F12), then enable + start deliberately:
  on a box rebuilt from the offseason snapshot, "deliberately" means the order of
  `docs/ops/2027-season-start.md` row B8c (old code, old crontab and the leaderboard timer are
  dealt with before anything is started).

## 5. Deploy / rollback commands

**Normal ship** (keeps `main` as the source of truth that gets deployed):

```bash
git checkout -b fix/<short-desc>
# make the fix + add/adjust tests
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
git commit -am "fix: <what and why>"
git push origin fix/<short-desc>     # open a PR so Eric can eyeball it from his phone
# after it's merged to main:
git push origin main:deploy          # ship — restarts the scheduler; read §1 on what the canary does not cover
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
> `protect-main-deploy` ruleset will reject it (see the break-glass note under blunt rollback).

**Revert a bad change** (preferred — history keeps moving forward):

```bash
git checkout main && git pull
git revert <bad-sha>
git push origin main            # keep main canonical
git push origin main:deploy     # ship the revert
```

**Blunt rollback** to a known-good SHA (stopgap — get prod healthy now, fix `main` after).
"Known-good" means a commit that has already served picks IN THE CURRENT SEASON: never a
commit from before the season's first deploy (the October 2026 code rewrites 2027 state the
current code then refuses, §1). The same holds for the revert flow above: revert the bad
change, do not go back past the season's first deploy.

```bash
git push origin <good-sha>:deploy --force   # workflow hard-resets the box to this ref
```

> 🛑 **Break-glass required (2026-07-10):** the `protect-main-deploy` ruleset blocks
> force-pushes and deletions on `main` + `deploy` for EVERYONE (no bypass actors) — a stray
> tool rewinding `deploy` would otherwise hard-reset production. To run this rollback:
> repo → Settings → Rules → Rulesets → `protect-main-deploy` → Enforcement **Disabled** →
> force-push → re-enable. 30 seconds, and a deliberate speed bump before hard-resetting prod.
> Prefer the revert flow above whenever there's time.

**Just restart services** off the current `deploy` ref, no code change: GitHub → Actions →
"Deploy to Hetzner" → **Run workflow** (`workflow_dispatch`).

## 6. Where signal comes from when you're remote

- **Bluesky DM health alerts** — CRITICAL alerts from `bts.health.runner` DM Eric; he'll relay
  the text, which usually names the failing subsystem.
- **GitHub Actions log** of the last deploy — canary failure reasons, in plain text.
- **The repo + tests** — reproduce code bugs here.
- You **cannot** see the live dashboard or box logs from the cloud. Don't guess at on-box
  state — ask Eric or escalate per §3.

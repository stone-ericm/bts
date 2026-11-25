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
- The workflow hard-resets to the run's tested SHA, runs `uv sync --locked --extra model`, and
  restarts both services. The canary accepts an active scheduler or a clean exit awaiting
  restart, an active dashboard unit and dashboard HTTP 2xx. A failed canary attempts reset to
  pre-deploy HEAD, dependency sync and another restart. Early fetch, reset, sync or restart
  failure can bypass rollback, and a passed canary can be followed by job failure. The canary
  does not exercise a live pick. After any failed job verify actual HEAD, sync and unit states
  via SSH. October `619abda7` is unsafe for 2027, so use `docs/ops/2027-season-start.md` row
  B8c for restored boxes. (Whole-range review W12, W13, W17: the workflow change is deferred to
  January 2027.)
- **Do NOT** `systemctl restart` on the box after pushing — the workflow already does it.
- `origin/main` and `origin/deploy` are **independent**. Deploy ships *whatever ref you push
  to `deploy`*. Normal path is `git push origin main:deploy`; you can also push a branch with
  `git push origin HEAD:deploy` (see the ⚠️ footgun in §5).

## 2. What you CAN do from a cloud session

- ✅ Diagnose a **code bug** against the repo + tests.
- ✅ Write a fix on a branch.
- ✅ **Ship it:** ship to `deploy` only in a verified scheduler sleep window with no lineup
  check, send, grading or entry check in flight or due during the deploy. Use B8c on an
  offseason restore. The canary checks the two services as described in §1; re-runs and
  dispatches restart both as well.
- ✅ **Roll back / revert** (§5).
- ✅ **Apply the current `deploy` SHA again:** GitHub → Actions → "Deploy to Hetzner" →
  **Run workflow** (`workflow_dispatch`) — **select the `deploy` branch in the dropdown**
  (defaults to `main`, which fails at the environment gate: the root SSH key lives in the
  `production` environment whose branch policy allows only `deploy`; rotated 2026-07-10).
  For a restart only, have the on-box operator restart the required units during the §2
  quiet window. Use dispatch when you intend to apply the current `deploy` SHA, never
  after a canary rollback that left `deploy` pointing at the failed commit.
- ✅ **Read the last deploy's Actions log:** read the Actions log for the stage and any canary
  reason; verify actual on-box state after failure.

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
| **Re-capture expired leaderboard cookies** (`scripts/capture_bts_cookies.py`) | Interactive on Eric's Mac. Copy the resulting cookie file to the box. |
| Inspect/repair on-box data under `/home/bts/...` | on the box |

**The cookie case is the big one.** If picks have frozen and the signal points at leaderboard
auth / login / cookies, that is **not a code bug**, and there is no cloud fix. Read the exact
account failure before acting. On RATE-LIMITED or an account-request 403/429, stop manual
retries and cookie capture and ask Eric when to resume. On transient, outage-shaped, network,
or cookie-refresh-not-indicated errors, wait for the next scheduled attempt; escalate repeated
failures for diagnosis before changing cookies. For `auth/cookie error — refresh via
capture_bts_cookies.py on Mac`, have Eric verify `~/.bts-leaderboard-cookies.json` on the box and
his MLB login on his Mac, and if absent or expired re-capture on his Mac and install the JSON
on the box (chmod 600). For shape, identity or ambiguous errors, preserve the diagnostic and
escalate without recapture. Example for the auth/cookie case:

> "The contest fetch reported an auth/cookie error. Eric, please check
> `~/.bts-leaderboard-cookies.json` on bts-hetzner and your MLB login on your Mac; if absent or
> expired, re-capture from the repo on your Mac with
> `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/capture_bts_cookies.py --print` (it stores
> the cookies in the Mac Keychain and also prints the JSON), and install that JSON as
> `~/.bts-leaderboard-cookies.json` for user `bts` on the box (chmod 600), keeping no other copy."

## 4. Playbook by symptom

**Triage rule:** *Can you reproduce or locate it in the repo + tests?* → fix forward (§2).
*Is it about the box's runtime, auth/cookies, config files, or network?* → you can't reach it
from the cloud (§3) → escalate.

- **A deploy failed / canary rolled back:** open Actions → last "Deploy to Hetzner" run. Use
  `Pre-deploy SHA` as the attempted rollback target only. Have the SSH operator verify and
  report the actual checked-out SHA, sync and service states to Eric before shipping a fix.
  Escalate an unhealthy or unverifiable rollback and any incompatible pre-deploy revision.
- **Picks frozen / none posted:** check the health DM Eric received and the Actions log. A code
  exception in a recent change → fix forward. Auth / cookie / box → **escalate** (§3).
- **Bad pick logic, service otherwise up:** code fix → redeploy, but **respect the leakage
  safety rules in `CLAUDE.md`** — never ship feature changes without the `shift(1)` guards +
  `scripts/leakage_audit.py`. A wrong-but-running model is not an emergency; don't make it
  worse with a rushed change.
- **Dashboard down, picks fine:** have the on-box operator run
  `systemctl --user restart bts-dashboard`. If using a deploy or dispatch, follow §2's
  quiet-window rule; both units restart and a failed canary can attempt a code rollback.
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
  `bash scripts/install-systemd-hetzner.sh` (audit F12), from a checkout of the approved commit. If using the offseason snapshot, apply
  B8c's hold before normal boot and follow B8c through step 4 so the approved 2027 commit is
  deployed with the scheduler stopped and cron and the leaderboard still held. Refresh standard
  unit files with `bash scripts/install-systemd-hetzner.sh` only after that deploy (re-applying
  any `/dev/null` mask on held October writer `.service` files before `daemon-reload`), then
  finish B8c steps 5–7.

## 5. Deploy / rollback commands

**Normal ship** (keeps `main` as the source of truth that gets deployed):

```bash
git checkout -b fix/<short-desc>
# make the fix + add/adjust tests
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
git commit -am "fix: <what and why>"
git push origin fix/<short-desc>     # open a PR so Eric can eyeball it from his phone
# after it's merged to main:
git push origin main:deploy          # ship in the §2 quiet window; see §1 for canary limits
```

**Emergency lane** (picks actively broken, fix is small + obvious, Eric ok'd it in chat) —
ship the branch straight to `deploy` (the §2 quiet-window rule still applies):

```bash
git push origin HEAD:deploy
```

> ⚠️ **Then land the same fix on `main`** (merge the PR). `deploy` and `main` are independent,
> so the next normal `main:deploy` will **revert** anything that's on `deploy` but not on
> `main`. An emergency branch-ship is not done until main has it too.
> **Use a true merge, not a squash**: a squash-merge means the emergency commit is not an
> ancestor of `main`, so the reconciling `main:deploy` push is non-fast-forward and the
> `protect-main-deploy` ruleset will reject it (see the break-glass note under blunt rollback).

**Revert a bad change** (preferred — history keeps moving forward; ship it in the §2 quiet
window, and the revert must retain the current-season guards):

```bash
git checkout main && git pull
git revert <bad-sha>
git push origin main            # keep main canonical
git push origin main:deploy     # ship the revert
```

**Blunt rollback** (stopgap). Use only an approved target verified compatible with the current
season and present state, and follow the §2 quiet-window rule. A commit that served this season
is only a candidate. If none qualifies, keep outward delivery held and fix forward. A revert
must retain the current-season guards.

```bash
git push origin <good-sha>:deploy --force   # workflow hard-resets the box to this ref
```

> 🛑 **Break-glass required (2026-07-10):** the `protect-main-deploy` ruleset blocks
> force-pushes and deletions on `main` + `deploy` for EVERYONE (no bypass actors) — a stray
> tool rewinding `deploy` would otherwise hard-reset production. To run this rollback:
> repo → Settings → Rules → Rulesets → `protect-main-deploy` → Enforcement **Disabled** →
> force-push → re-enable. 30 seconds, and a deliberate speed bump before hard-resetting prod.
> Prefer the revert flow above whenever there's time.

**Just restart services:** for a restart only, have the on-box operator restart the required
units during the §2 quiet window. Use dispatch when you intend to apply the current `deploy`
SHA (GitHub → Actions → "Deploy to Hetzner" → **Run workflow**, in the §2 quiet window): both
services restart and the workflow may roll back. Never dispatch after a canary rollback that left
`deploy` pointing at the failed commit.

## 6. Where signal comes from when you're remote

- **Bluesky DM health alerts** — CRITICAL alerts from `bts.health.runner` DM Eric; he'll relay
  the text, which usually names the failing subsystem.
- **GitHub Actions log** of the last deploy — canary failure reasons, in plain text.
- **The repo + tests** — reproduce code bugs here.
- You **cannot** see the live dashboard or box logs from the cloud. Don't guess at on-box
  state — ask Eric or escalate per §3.
